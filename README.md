# ERP minimizer

In this project, we aim to minimize the Energy-Response Product (ERP) for a machine learning workload deployed on k servers. 

In the `analysis` folder, you can find the analysis of the data that we collected, where, in various Jupyter notebooks, we plot and review the data. Furthermore, we also create predictive models using this data here.

The `optimizer` directory on the other hand, contains the final optimizers as `joblib` files and a script to optimize the ERP for a prespecified workload (`experiment_configs.csv`).
