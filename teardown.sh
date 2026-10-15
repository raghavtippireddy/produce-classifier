#!/usr/bin/env bash
# Delete everything deploy.sh created. Leaves the budget alert in place.
set -uo pipefail

NAME=produce-classifier
REGION=${AWS_REGION:-$(aws configure get region || true)}
REGION=${REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="$NAME-$ACCOUNT-$REGION"
export AWS_REGION=$REGION AWS_DEFAULT_REGION=$REGION

aws lambda delete-function-url-config --function-name "$NAME" 2>/dev/null && echo "deleted function URL"
aws lambda delete-function --function-name "$NAME" 2>/dev/null && echo "deleted Lambda"
aws logs delete-log-group --log-group-name "/aws/lambda/$NAME" 2>/dev/null && echo "deleted logs"
aws iam delete-role-policy --role-name "$NAME-role" --policy-name read-model 2>/dev/null
aws iam detach-role-policy --role-name "$NAME-role" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null
aws iam delete-role --role-name "$NAME-role" 2>/dev/null && echo "deleted IAM role"
aws s3 rb "s3://$BUCKET" --force >/dev/null 2>&1 && echo "deleted bucket $BUCKET"
echo "done"
