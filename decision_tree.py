import re
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
# data engineering
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, OrdinalEncoder, LabelEncoder
# decision tree modelling
from sklearn.tree import DecisionTreeRegressor


def df_encoded_overall(encoded_features, columns_encode, df_columns, ohe_x):

    # prepare columns names for encoded features
    toencode_colnames = df_columns[columns_encode]
    ohe = ohe_x.transformers_[0][1]

    # get feature names for encoded features
    encoded_colnames = ohe.get_feature_names()

    # set columns names for encoded features
    for i in reversed(range(columns_encode.sum())):
        column_name = "x" + str(i)
        encoded_colnames = [re.sub(column_name, toencode_colnames[i], x) for x in encoded_colnames]

    # set columns names for final dataset output
    df_features_encoded = pd.DataFrame(encoded_features, columns=encoded_colnames + ['cc_action'])

    df_overall = df_features_encoded.copy()

    return df_overall


def get_dt_model(orig_df: pd.DataFrame):
    orig_df_copy = orig_df[orig_df["cc_action"].notnull()].copy()

    columns_encode = orig_df_copy.columns != 'cc_action'
    # Encoder is turning all dataset into numbers. One-Hot-Encode features, and turn target labels into ordinal numbers
    ohe_x = make_column_transformer(
        (OneHotEncoder(sparse=False, handle_unknown="ignore"), columns_encode),
        (OrdinalEncoder(categories=[["Reject Offer", "No Contact", "Accept Offer"]]), ['cc_action']))

    prepare_df_transformer = FunctionTransformer(df_encoded_overall, validate=False,
                                                 kw_args={"columns_encode": columns_encode,
                                                          "df_columns": orig_df_copy.columns,
                                                          "ohe_x": ohe_x})  # don't report errors
    pipeline = Pipeline([
        ('column_transform', ohe_x),
        ('prepare_df', prepare_df_transformer)
    ])

    # get final dataframe ready for Decision Tree Model use
    dftemp = orig_df_copy.copy()

    df_trans = pipeline.fit_transform(dftemp)

    target = 'cc_action'

    X = df_trans.loc[:, (df_trans.columns != target)]
    y = df_trans[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    dt = DecisionTreeRegressor()

    # range of possible parameters
    params_dt = {'max_depth': [2, 3, 4, 5, 6, 7, 8], 'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]}

    # Instantiate grid_dt
    grid_dt = GridSearchCV(estimator=dt,
                           param_grid=params_dt,
                           scoring='neg_mean_squared_error',
                           cv=5,
                           n_jobs=-1,
                           iid=False)

    # Select best performing decision tree among range of parameters inputs
    grid_dt.fit(X_train, y_train)

    best_model = grid_dt.best_estimator_

    predictions = best_model.predict(X_test)

    # This model score should be logged down in production to monitor performance of model.
    current_model_score = mean_squared_error(y_test, predictions)
    print(current_model_score)

    return pipeline, best_model


def add_likelihood_from_dt(orig_df: pd.DataFrame, pipeline, best_model, rules_var_values):
    orig_df_copy = orig_df[orig_df["cc_action"].isnull()].copy()
    columns_encode = orig_df_copy.columns != 'cc_action'
    df_trans = orig_df_copy.copy()

    df_trans = df_trans.dropna(how="all")
    df_trans = df_trans.drop(["cc_action"], axis=1)

    columns_encode_nolabel = (df_trans.columns != 'ideal_location')
    toencode_colnames = df_trans.columns[columns_encode_nolabel]

    transf = pipeline.named_steps['column_transform'].transformers_
    ohe_nolabel = transf[0][1]

    encoded_dataset = ohe_nolabel.transform(df_trans)

    x_unseen_data = df_encoded_overall_nolabel(encoded_dataset, ohe_nolabel, columns_encode, toencode_colnames)
    encoded_columns = x_unseen_data.columns

    for col in orig_df.columns.drop(["cc_action"]):
        x_col_names = [X_col for X_col in x_unseen_data.columns if col in X_col]
        x_unseen_data[col] = x_unseen_data[x_col_names].idxmax(1).str.replace(f"{col}_", "")

    x_unseen_data_wo_encoded = x_unseen_data.copy()
    x_unseen_data_wo_encoded = x_unseen_data_wo_encoded.drop(encoded_columns, axis=1)
    data_for_ga = pd.DataFrame({'study_level': rules_var_values[:, 0],
                                'acceptable_distance': rules_var_values[:, 1],
                                'acceptable_fees': rules_var_values[:, 2],
                                'second_language': rules_var_values[:, 3],
                                'dietary_restrictions': rules_var_values[:, 4],
                                'service_type': rules_var_values[:, 5],
                                'childcare_rank': rules_var_values[:, 6],
                                'enrol_reg_diff': rules_var_values[:, 7],
                                'today_reg_diff': rules_var_values[:, 8]})
    data_for_ga["predictions"] = 2
    data_ga_predicted = x_unseen_data_wo_encoded.merge(data_for_ga,
                                                       on=['study_level', 'acceptable_distance',
                                                           'acceptable_fees', 'second_language',
                                                           'dietary_restrictions', 'service_type',
                                                           'childcare_rank', 'enrol_reg_diff',
                                                           'today_reg_diff'],
                                                       how="left")
    data_ga_predicted = data_ga_predicted[data_ga_predicted["predictions"].notnull()]
    x_unseen_data = x_unseen_data.merge(data_ga_predicted,
                                        on=['study_level', 'acceptable_distance',
                                            'acceptable_fees', 'second_language',
                                            'dietary_restrictions', 'service_type',
                                            'childcare_rank', 'enrol_reg_diff',
                                            'today_reg_diff'],
                                        how="left")
    x_unseen_data_w_predictions = x_unseen_data[x_unseen_data["predictions"].notnull()]
    x_unseen_data_wo_predictions = x_unseen_data[x_unseen_data["predictions"].isnull()]
    x_unseen_data_wo_predictions["predictions"] = best_model.predict(x_unseen_data_wo_predictions[encoded_columns])
    x_unseen_data = pd.concat([x_unseen_data_wo_predictions, x_unseen_data_w_predictions],
                              axis=0).drop_duplicates()
    x_unseen_data = x_unseen_data.drop(encoded_columns, axis=1)

    return x_unseen_data


def df_encoded_overall_nolabel(encoded_features, ohe_NoLabel, columns_encode, toencode_colnames):
    # Use previously fitted one-hot-encoder
    ohe = ohe_NoLabel
    # get feature names for encoded features
    encoded_colnames = ohe.get_feature_names()

    # set columns names for encoded features
    for i in reversed(range(columns_encode.sum())):
        column_name = "x" + str(i)
        encoded_colnames = [re.sub(column_name, toencode_colnames[i], x) for x in encoded_colnames]

    # set columns names for final dataset output
    df_features_encoded = pd.DataFrame(encoded_features, columns=encoded_colnames)

    df_overall = df_features_encoded.copy()

    return df_overall

