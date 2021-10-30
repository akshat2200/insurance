

# =======================================
# Import Packages
# =======================================
from required import *

input_dest, input_req, out_dest = config_fetch()

def main():

    global out_dest
    out_dir = out_dest + '\Output'
    output_path = Path(out_dir)
    now = datetime.now()

    insc_data = load_dataframe()

    print("Raw data pre-processing started...")
    insc_dff = data_prep(insc_data)
    print("Pre-processing completed")

    print("Running Trained Regression Model on Insurance Claim Data...")
    insc_out = predict_func(insc_dff)
    print("Prediction on data completed")

    print("Started saving output file...")
    insc_out.to_excel(str(output_path) + r"/Incurance Ultimate Claim - Data Load" + now.strftime("%d%B%y") + ".xlsx", index = False, header = True)
    print("Run Successfully Completed")


# ===========================================
# Load raw dataframe for Insurance Claim
# ===========================================
def load_dataframe():

    global input_req
    dir_ = input_req + '\Input'
    file_path = Path(dir_)

    ls_files = glob.glob(str(file_path) + r"/*.xlsx")
    latest_file = max(ls_files, key = os.path.getctime)
    file_name = os.path.basename(latest_file)

    print(f'Loading Raw DataFrame from location: {file_name}')
    insc_data = pd.read_excel(latest_file).sort_values(by = ['Claim Number'], ascending = False)
    print("Raw data loaded")
    print(f"Raw data has {insc_data.shape[0]} number of claims")

    return insc_data

# ===========================================
# Data Preprocessing steps
# ===========================================
def data_prep(dff):

    dff['Month'] = pd.DatetimeIndex(dff['date_of_loss']).month
    dff['Weekend'] = np.where((pd.to_datetime(dff['date_of_loss']).dt.dayofweek) > 4, 1, 0)

    dff['MSL'] = round(dff['Inception_to_loss']/30, 0) # Months since loss

    dff.drop(["Incurred","date_of_loss",'Loss_code','Loss_description','TP_type_insd_pass_front', 'TP_type_pass_multi'], axis = 1, inplace=True, errors="ignore")

    return dff

# ===========================================
# Run Trained Prediction Model
# ===========================================
def predict_func(dff):

    global input_req

    # Load trained regression model
    print("Loading trained regression model...")
    dir_pipe_predict = input_req + '\\Code\\model.pkl'
    with open(dir_pipe_predict, "rb") as f:
        pipeline_predict = pickle.load(f)
    print("Trained regression model loaded")

    # Load trained traget column encoder
    print("Loading trained target column encoder")
    dir_pipe_target = input_req + '\\Code\\target.plk'
    with open(dir_pipe_target, "rb") as f:
        pipeline_target = pickle.load(f)
    print("Trained target encoder loaded")

    features_testdata = dff.drop(['Incurred', 'Claim Number', 'Incurred_cat'], axis = 1, inplace=True, errors="ignore")
    id_col_testdata = dff['Claim Number']

    print("Predicting Claim amount on dataset...")
    predictions_test = pipeline_predict.predict(features_testdata)
    print("Prediction completed")
    y_pred = pd.DataFrame(pipeline_target.inverse_transform(predictions_test.reshape(-1,1)), columns = ["predictions"])

    y_predictions = pd.concat([id_col_testdata, y_pred], axis = 1)

    return y_predictions


if __name__ == "__main__":
    main()
