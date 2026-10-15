#!/usr/bin/env bash
# Deploy (or redeploy) the produce classifier: S3 bucket for the model, IAM role,
# Lambda function, and a public Function URL. Safe to re-run.
# Optional: BUDGET_EMAIL=you@example.com ./deploy.sh  also creates a $5/month budget alert.
set -euo pipefail
cd "$(dirname "$0")"

NAME=produce-classifier
REGION=${AWS_REGION:-$(aws configure get region || true)}
REGION=${REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="$NAME-$ACCOUNT-$REGION"
ROLE="$NAME-role"
export AWS_REGION=$REGION AWS_DEFAULT_REGION=$REGION
echo "Account $ACCOUNT, region $REGION"

[ -f model/model.tflite ] || { echo "model/model.tflite missing - run train.py first"; exit 1; }

echo "== 1/5 build Lambda package (Linux arm64, Python 3.12)"
rm -rf build && mkdir -p build/pkg
pip install -q --target build/pkg --implementation cp \
  --platform manylinux_2_27_aarch64 --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 \
  --python-version 3.12 --only-binary=:all: --upgrade -r lambda/requirements.txt
cp lambda/handler.py lambda/index.html build/pkg/
(cd build/pkg && zip -qr9 ../function.zip . -x '*/__pycache__/*')
echo "   function.zip: $(du -h build/function.zip | cut -f1)"

echo "== 2/5 S3 bucket (private) + model upload"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = us-east-1 ]; then aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
fi
aws s3 cp model/model.tflite "s3://$BUCKET/model/model.tflite" --only-show-errors
aws s3 cp model/labels.json "s3://$BUCKET/model/labels.json" --only-show-errors

echo "== 3/5 IAM role (logs + read this bucket only)"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  NEW_ROLE=1
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name read-model --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::$BUCKET/model/*\"}]}"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)
[ "${NEW_ROLE:-}" ] && { echo "   waiting 15s for the new role to propagate"; sleep 15; }

echo "== 4/5 Lambda function"
ENV="Variables={MODEL_BUCKET=$BUCKET,MODEL_PREFIX=model/}"
if aws lambda get-function --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$NAME" --zip-file fileb://build/function.zip >/dev/null
  aws lambda wait function-updated --function-name "$NAME"
  aws lambda update-function-configuration --function-name "$NAME" --environment "$ENV" >/dev/null
else
  aws lambda create-function --function-name "$NAME" --runtime python3.12 --architectures arm64 \
    --handler handler.lambda_handler --role "$ROLE_ARN" --zip-file fileb://build/function.zip \
    --memory-size 1024 --timeout 30 --environment "$ENV" >/dev/null
fi
aws lambda wait function-active-v2 --function-name "$NAME"
aws lambda wait function-updated --function-name "$NAME"

echo "== 5/5 public Function URL"
if ! aws lambda get-function-url-config --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$NAME" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$NAME" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
  # Newer accounts also require lambda:InvokeFunction for public URLs.
  aws lambda add-permission --function-name "$NAME" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >/dev/null 2>&1 || true
fi
URL=$(aws lambda get-function-url-config --function-name "$NAME" --query FunctionUrl --output text)

if [ -n "${BUDGET_EMAIL:-}" ]; then
  aws budgets create-budget --account-id "$ACCOUNT" \
    --budget '{"BudgetName":"monthly-5usd","BudgetLimit":{"Amount":"5","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
    --notifications-with-subscribers "[{\"Notification\":{\"NotificationType\":\"ACTUAL\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":80},\"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$BUDGET_EMAIL\"}]}]" \
    2>/dev/null && echo "   budget alert created (emails $BUDGET_EMAIL at \$4)" || echo "   budget already exists, skipped"
fi

echo
echo "Live at: $URL"
