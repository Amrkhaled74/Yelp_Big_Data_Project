from datetime import datetime
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.task_group import TaskGroup
from airflow.operators.empty import EmptyOperator

# --- Pipeline dependency mappings and schema topologies ---
# defines prerequisite dimensions needed for fact foreign/surrogate key resolution
FACT_DEPS = {
    "fact_tips" : ["dim_date","dim_business","dim_user","dim_tip"], 
    "fact_review" : ["dim_date","dim_business","dim_user","dim_review"] 
}

# defines hierarchy and parent dimension dependencies for relationship bridge tables
DIM_DEPS  = {

    "dim_business_hours_bridge" : ["dim_business_hours","dim_business"], 
    "dim_category_subcategory_bridge" : ["dim_category","dim_subcategory"],
    "dim_business_subcategory_bridge" :["dim_business","dim_subcategory"],
    "dim_user_friend" :["dim_user"],
}

# raw yelp JSON dataset extract endpoints
SOURCES = ["user", "tip", "review", "checkin", "business"]

# derives unique set of all dimensions, parent lookups, and bridge tables in alphabetical order
DIMS = sorted({d for F_deps in FACT_DEPS.values() for d in F_deps  }
              |set(DIM_DEPS)
              |{s for d_deps in DIM_DEPS.values() for s in d_deps  }
              )

# --- Pipeline DAG orchestration ---
with DAG(
    dag_id="yelp_pyspark_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,                  # Only 1 DAG run at any given moment
    # limits local Windows machine to 1 active Spark job to prevent driver OOM
    max_active_tasks=1,                 # Max 4 concurrent tasks across all active runs
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end", trigger_rule="all_success")

    # ---- Staging ----------------------------------------------------------
    # executes external Windows shell scripts over SSH to stage raw source extracts
    with TaskGroup('staging') as staging:
        stg = {
           s:SSHOperator(
            task_id=F"stg_{s}",
            ssh_conn_id="my_windows_host",
            command=f'cmd /c "C:\\scripts\\stg_{s}.bat"',
            cmd_timeout=3600,
            conn_timeout=60,
                
           )
           for s in SOURCES


        }

            # ---- dimensions ----------------------------------------------------------
    # executes batch scripts to process and persist silver dimension tables
    with TaskGroup('dimensions') as dimensions:
        dim = {
           d:SSHOperator(
            task_id=F"load_{d}",
            ssh_conn_id="my_windows_host",
            command=f'cmd /c "C:\\scripts\\{d}.bat"',
            cmd_timeout=3600,
            conn_timeout=60,
                
           )
           for d in DIMS


        }
        

            # ---- fact ----------------------------------------------------------
    # executes batch scripts to build star schema fact tables after dimensions materialize
    with TaskGroup('facts') as facts:
        fact = {
           f:SSHOperator(
            task_id=F"load_{f}",
            ssh_conn_id="my_windows_host",
            command=f'cmd /c "C:\\scripts\\{f}.bat"',
            cmd_timeout=3600,
            conn_timeout=60,
                
           )
           for f in FACT_DEPS


        }
        
    
    
    # ---- Wiring -----------------------------------------------------------
    # enforces strict architectural progression from Bronze staging to Silver dimensions
    start >> staging >> dimensions  # >> dimensions
       # Dimension -> dimension (parents load before children)
    # wires prerequisite parent dimensions prior to dependent bridge tables
    for child, parents in DIM_DEPS.items():
        for p in parents:
            dim[p] >> dim[child]


    # wires dimension dependencies before fact tables can execute surrogate lookups
    for f, deps in FACT_DEPS.items():
        for d in deps:
            dim[d] >> fact[f]
     

    list(fact.values()) >> end 
        
    