import os
import pandas as pd
import re
import joblib
import json
from sklearn.ensemble import RandomForestClassifier

def load_dataset(path):
    # Take the set of files and read them all into a single pandas dataframe
    files = [ os.path.join(path, file) for file in os.listdir(path) ]
    
    if len(files) == 0:
        raise ValueError("Invalid # of files in dir: {}".format(path))

    raw_data = [ pd.read_csv(file, sep=",", header=None ) for file in files ]
    data = pd.concat(raw_data)

    # labels are in the first column
    y = data.iloc[:,0]
    X = data.iloc[:,1:]
    return X,y
    
def start(args):
    print("Training mode")

    try:
        X_train, y_train = load_dataset(args.train)
        X_test, y_test = load_dataset(args.validation)
        
        hyperparameters = {
            "max_depth": args.max_depth,
            "verbose": 1, # show all logs
            "n_jobs": args.n_jobs,
            "n_estimators": args.n_estimators
        }
        print("Training the classifier")
        model = RandomForestClassifier()
        model.set_params(**hyperparameters)
        model.fit(X_train, y_train)
        print("Score: {}".format( model.score(X_test, y_test)) )
        joblib.dump(model, open(os.path.join(args.model_dir, "iris_model.pkl"), "wb"))
    
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, "failure"), "w") as s:
            s.write("Exception during training: " + str(e) + "\\n" + trc)
            
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during training: " + str(e) + "\\n" + trc, file=sys.stderr)
        
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
