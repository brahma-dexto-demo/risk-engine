# Staging deployment

All AWS calls below run inside the managed AWS MCP Server's `aws___run_script`
with boto3, under `DextoDemoRole`, in `us-east-1`. The deployer's computer has no
AWS credentials, AWS CLI, or kubectl. CodeBuild's build container uses its own
service role and AWS CLI for the ECR login in `buildspec.yml`; those commands are
not deployer commands. Account IDs are always resolved with STS at deployment time.
Infrastructure (buckets, ECR repositories, roles, project source configuration,
networking, and log groups) is provisioned by `brahma-dexto-demo/demo-infra`.
CodeBuild sources must point to the published repository and accept the supplied
commit SHA; these local repositories have not been pushed.

1. Commit the model, inputs, lockfile, and code. Start `risk-engine-image` at the
   full SHA using `aws___run_script`, and wait for success:

```python
import boto3
import time

session = boto3.Session(region_name="us-east-1")
account = session.client("sts").get_caller_identity()["Account"]
sha = "<sha>"  # Full committed source revision
codebuild = session.client("codebuild")
build_id = codebuild.start_build(
    projectName="risk-engine-image",
    sourceVersion=sha,
    environmentVariablesOverride=[
        {
            "name": "ECR_URI",
            "value": f"{account}.dkr.ecr.us-east-1.amazonaws.com/brahma-demo/risk-engine",
            "type": "PLAINTEXT",
        }
    ],
)["build"]["id"]
# For long builds, return build_id and poll in subsequent run_script calls.
for _ in range(60):
    build = codebuild.batch_get_builds(ids=[build_id])["builds"][0]
    status = build["buildStatus"]
    if status == "SUCCEEDED":
        break
    if status in {"FAILED", "FAULT", "STOPPED", "TIMED_OUT"}:
        raise RuntimeError(f"Build {build_id}: {status}; inspect build['logs']")
    time.sleep(10)
else:
    raise TimeoutError(f"Build {build_id} is still running; resume polling")
image = f"{account}.dkr.ecr.us-east-1.amazonaws.com/brahma-demo/risk-engine:{sha}"
print({"image": image, "account": account})
```

2. Confirm `accounts/accounts.json` is seeded in `brahma-demo-data-<account>`
   (the same bytes as accounts-api). Read `batch/job-definition.json` locally,
   replace `RISK_ENGINE_IMAGE` with the built image and every `ACCOUNT_ID` with the
   STS account. Supply the rendered JSON object to `run_script` as `rendered`:

```python
batch = session.client("batch")
registered = batch.register_job_definition(**rendered)
job_id = batch.submit_job(
    jobName=f"risk-engine-{sha[:12]}",
    jobQueue="brahma-demo-queue",
    jobDefinition=registered["jobDefinitionArn"],
    tags={"project": "brahma-demo"},
    propagateTags=True,
)["jobId"]
print({"job_id": job_id})
```

Preserve `tags={"project": "brahma-demo"}` and `propagateTags=true` in the rendered
definition and submission so job tags reach the ECS task, as demo-infra requires.

The definition uses Fargate with 1 vCPU / 2 GB, roles `brahma-demo-batch-exec` and
`brahma-demo-batch-job`, and `/aws/batch/job` logs. The execution role pulls ECR
and writes logs; the job role reads account input and writes scores. The demo
queue's public subnets must support assigned public IPs (or adjust to private
subnets with NAT/endpoints in demo-infra).

3. Poll the job and read its logs via AWS APIs. Resume polling in separate
   `run_script` calls if the MCP execution time limit is shorter than the job.

```python
for _ in range(120):
    job = batch.describe_jobs(jobs=[job_id])["jobs"][0]
    if job["status"] in {"SUCCEEDED", "FAILED"}:
        break
    time.sleep(5)
else:
    raise TimeoutError(f"Job {job_id} still running; resume polling")
stream = job.get("container", {}).get("logStreamName")
if stream:
    logs = session.client("logs")
    token = None
    while True:
        request = {"logGroupName": "/aws/batch/job", "logStreamName": stream, "startFromHead": True}
        if token:
            request["nextToken"] = token
        page = logs.get_log_events(**request)
        for event in page["events"]:
            print(event["message"])
        new_token = page["nextForwardToken"]
        if new_token == token:
            break
        token = new_token
if job["status"] != "SUCCEEDED":
    raise RuntimeError(job.get("statusReason", "Batch failed; inspect container exitCode/reason"))
```

4. Check the printed evaluation metrics passed, then read the score artifact:

```python
import json

s3 = session.client("s3")
result = json.loads(
    s3.get_object(Bucket=f"brahma-demo-data-{account}", Key="scores/latest.json")["Body"].read()
)
print(
    {
        "model_version": result["model_version"],
        "generated_at": result["generated_at"],
        "accounts_scored": len(result["scores"]),
    }
)
```

Check `generated_at` belongs to this run: the output key is overwritten on each
score invocation. A failed eval marks the job failed even if scoring wrote a file.

## Local run

```sh
docker build -t risk-engine:local .
docker run --rm risk-engine:local
```

To retain the result locally, mount a writable data directory (allow UID 10001 to
write it): `docker run --rm -e LOCAL_DATA_DIR=/data -v "$PWD/data:/data" risk-engine:local`.
