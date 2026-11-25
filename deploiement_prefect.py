from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow, code_flow

if __name__ == "__main__":
    d_all = all_flow.to_deployment(name="ml-pipeline-all", cron="0 2 * * *")
    d_train = train_flow.to_deployment(name="ml-pipeline-train")
    d_eval = evaluate_flow.to_deployment(name="ml-pipeline-evaluate")
    d_code = code_flow.to_deployment(name="ml-pipeline-code")
    serve(d_all, d_train, d_eval, d_code)
