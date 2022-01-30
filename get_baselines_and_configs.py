# #####################################################################################################################
#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.                                                 #
#                                                                                                                     #
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance     #
#  with the License. A copy of the License is located at                                                              #
#                                                                                                                     #
#  http://www.apache.org/licenses/LICENSE-2.0                                                                         #
#                                                                                                                     #
#  or in the 'license' file accompanying this file. This file is distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES  #
#  OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions     #
#  and limitations under the License.                                                                                 #
# #####################################################################################################################
import os
import boto3
import argparse
import logging
from utils import (
    exception_handler,
    read_config_from_json,
    get_baselines_and_model_name,
    process_bias_baselines,
    process_explainability_config_file,
    get_built_in_model_monitor_image_uri,
    extend_config,
    write_config_to_json,
)

# create clients
logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")


@exception_handler
def main():
    # define arguments
    parser = argparse.ArgumentParser("Get the arguments to create the data baseline job.")
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("LOGLEVEL", "INFO").upper(),
        help="Log level. One of ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']. Default 'INFO'.",
    )
    parser.add_argument(
        "--model-monitor-role",
        type=str,
        required=True,
        help="The AWS IAM execution role's arn used by the model monitor.",
    )
    parser.add_argument("--sagemaker-project-id", type=str, required=True, help="The AWS SageMaker project's id.")
    parser.add_argument("--sagemaker-project-name", type=str, required=True, help="The AWS SageMaker project's name.")
    parser.add_argument("--sagemaker-project-arn", type=str, required=True, help="The AWS SageMaker project's ARN.")
    parser.add_argument(
        "--monitor-outputs-bucket",
        type=str,
        required=True,
        help=(
            "Amazon S3 bucket that will be used to store the outputs of the "
            "SageMaker Model Monitor's Baseline and monitoring schedule jobs."
        ),
    )
    parser.add_argument(
        "--import-staging-config",
        type=str,
        default="staging-monitoring-schedule-config.json",
        help=(
            "The JSON file's name containing the monitoring schedule's staging configuration."
            "Default 'staging-monitoring-schedule-config.json'."
        ),
    )
    parser.add_argument(
        "--import-prod-config",
        type=str,
        default="prod-monitoring-schedule-config.json",
        help=(
            "The JSON file's name containing the monitoring schedule's prod configuration."
            "Default 'prod-monitoring-schedule-config.json'."
        ),
    )
    parser.add_argument(
        "--export-staging-config",
        type=str,
        default="staging-monitoring-schedule-config-export.json",
        help=(
            "The JSON file's name used to export the monitoring schedule's staging configuration."
            "Default 'staging-monitoring-schedule-config-export.json'."
        ),
    )
    parser.add_argument(
        "--export-prod-config",
        type=str,
        default="prod-monitoring-schedule-config-export.json",
        help=(
            "The JSON file's name used to export the monitoring schedule's prod configuration."
            "Default 'prod-monitoring-schedule-config-export.json'."
        ),
    )

    # parse arguments
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    # get the name of the S3 bucket used to store the outputs of the Model Monitor's
    monitor_outputs_bucket = args.monitor_outputs_bucket

    # use the endpoint name, deployed in staging env., to get baselines (from MR) and model name
    staging_config = read_config_from_json(args.import_staging_config)
    endpoint_name = f"{args.sagemaker_project_name}-{staging_config['Parameters']['StageName']}"
    baselines = get_baselines_and_model_name(endpoint_name, sm_client)
    logger.info("Baselines returned from MR, and Model Name...")
    logger.info(baselines)

    # update Bias and Explainability baselines
    updated_baselines = {
        "Bias": process_bias_baselines(baselines["DriftCheckBaselines"]["Bias"], s3_client),
        "Explainability": process_explainability_config_file(
            baselines["DriftCheckBaselines"]["Explainability"], baselines["ModelName"], s3_client
        ),
        "ModelQuality": baselines["DriftCheckBaselines"]["ModelQuality"],
        "ModelDataQuality": baselines["DriftCheckBaselines"]["ModelDataQuality"],
    }
    logger.info("Updated Baselines...")
    logger.info(updated_baselines)

    # get the ImageUri for model monitor and clarify
    monitor_image_uri = get_built_in_model_monitor_image_uri(
        region=boto3.session.Session().region_name, framework="model-monitor"
    )
    clarify_image_uri = get_built_in_model_monitor_image_uri(
        region=boto3.session.Session().region_name, framework="clarify"
    )

    # extend monitoring schedule configs
    logger.info("Update Monitoring Schedule configs for staging/prod...")
    staging_monitor_config = extend_config(
        args, monitor_image_uri, clarify_image_uri, updated_baselines, monitor_outputs_bucket, staging_config, sm_client
    )
    prod_monitor_config = extend_config(
        args,
        monitor_image_uri,
        clarify_image_uri,
        updated_baselines,
        monitor_outputs_bucket,
        read_config_from_json(args.import_prod_config),
        sm_client,
    )

    # export monitor configs
    logger.info("Export Monitoring Schedule configs for staging/prod...")
    write_config_to_json(args.export_staging_config, staging_monitor_config)
    write_config_to_json(args.export_prod_config, prod_monitor_config)


if __name__ == "__main__":
    main()
