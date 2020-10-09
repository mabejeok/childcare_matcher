import json
import pandas as pd

from datetime import datetime
from flask import Flask, render_template, request, url_for, session
from flask_mysqldb import MySQL
from sqlalchemy import create_engine
from werkzeug.utils import redirect

from decision_tree import get_dt_model, add_likelihood_from_dt
from ga import get_top_ga_solutions

TODAY = datetime.today()

app = Flask(__name__)
app.secret_key = "thisismysecretkey"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['MYSQL_USER'] = 'iss-user'
app.config['MYSQL_PASSWORD'] = 'iss-user'
app.config['MYSQL_DB'] = 'cc_schema'

mysql = MySQL(app)

db_connection_str = (f"mysql+pymysql://{app.config['MYSQL_USER']}" 
                     f":{app.config['MYSQL_PASSWORD']}@{app.config['MYSQL_HOST']}/{app.config['MYSQL_DB']}")
db_connection = create_engine(db_connection_str)

pd.set_option("display.max_rows", None, "display.max_columns", None)


@app.route('/')
def start_app():
    """
    Entry point for this app.
    :return: HTML template
    """
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    """
    Use the username entered by the user, go back to the database to check for password
    If user does not exist or password is not correct, then deny entry
    Else we batch predict likelihood for entries in db that has no likelihood (if we have sufficient
     past entries) and then direct user to home()
    :return: redirect to home() if successful, else return to login.
    """
    error = None
    if request.method == 'POST':
        df = pd.read_sql(f'SELECT login_key, password, centre_code, user_name '
                         f'FROM childcare_login '
                         f'WHERE user_name="{request.form.get("username")}"', con=db_connection)
        if df.empty:
            error = 'Invalid Credentials. Please try again.'
        else:
            entered_password = request.form.get("password")
            correct_password = df.iloc[0]["password"]
            if entered_password != correct_password:
                error = 'Invalid Credentials. Please try again.'
            else:
                session["loggedin"] = True
                session["id"] = str(df.iloc[0]["login_key"])
                session["username"] = df.iloc[0]["user_name"]
                centre_code = df.iloc[0]["centre_code"]
                batch_predict_likelihood(centre_code)
                return redirect(url_for('home', centre_code=centre_code))

        return render_template('login.html', error=error)


@app.route('/home/<centre_code>')
def home(centre_code):
    if "loggedin" in session:
        # Grab all entries related to the centre_code
        rank_df = pd.read_sql(f'SELECT parent_choice_key, child_idno, childcare_rank, '
                              f'reg_date, likelihood, cc_action '
                              f'FROM parent_choices '
                              f'WHERE centre_code="{centre_code}"', con=db_connection)
        child_idnos = rank_df["child_idno"].to_list()
        formatted_child_idnos = "','".join(child_idnos)
        child_details = pd.read_sql(f"SELECT child_idno, parent_contact, enrolment_date, study_level "
                                    f"FROM parent_details "
                                    f"WHERE child_idno IN ('{formatted_child_idnos}')", con=db_connection)
        full_result = child_details.merge(rank_df, on="child_idno", how="outer")
        # Likelihood is at first just number, need to convert to category High, Medium and Low.
        full_result['likelihood'] = full_result['likelihood'].map(lambda x: format_likelihood(x))
        # For any likelihood that is still empty e.g., if past entries are less than 50, then
        # use fuzzy logic and rules to guess the likelihood.
        full_result['likelihood'] = full_result.apply(lambda x: get_probability(x['enrolment_date'],
                                                                                x['reg_date'],
                                                                                x['childcare_rank'])
                                                      if x['likelihood'] is None and x['cc_action'] is None
                                                      else x['likelihood'],
                                                      axis=1)
        full_result['enrolment_date'] = full_result['enrolment_date'].apply(lambda x: x.strftime('%b-%Y'))
        full_result['reg_date'] = full_result['reg_date'].apply(lambda x: x.strftime('%d-%b-%Y'))
        full_result = full_result.drop(["childcare_rank"], axis=1)
        result = full_result.to_dict('records')
        return render_template('home.html', centre_code=centre_code, result=json.dumps(result))
    else:
        return redirect(url_for("start_app"))


@app.route("/logout")
def logout():
    session.pop("loggedin", None)
    session.pop("id", None)
    session.pop("username", None)
    mysql.connection.close()
    return redirect(url_for("start_app"))


@app.route("/update_data", methods=["POST"])
def update_data():
    """
    Grab information from select dropdowns from table in home.html and update mysql
    :return:
    """
    for key, value in request.json.items():
        cur = mysql.connection.cursor()
        cur.execute(f"UPDATE parent_choices "
                    f"SET cc_action = '{value}' "
                    f"WHERE parent_choice_key={key};")
        mysql.connection.commit()

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}


def format_likelihood(orig_likelihood):
    """
    Convert float to category
    :param orig_likelihood: Float number in string form
    :return: Category
    """
    if orig_likelihood is None:
        return orig_likelihood
    else:
        orig_likelihood = orig_likelihood.replace("\r", "")
        if orig_likelihood is not None:
            if float(orig_likelihood) < 1:
                return "Low"
            elif 1 <= float(orig_likelihood) < 2:
                return "Medium"
            else:
                return "High"
        else:
            return None


def get_probability(enrolment_date: datetime, reg_date: datetime, rank: int) -> float:
    """
    Here we use fuzzy logic and rules to guess the likelihood. The presumption is
    these conditions will ALL be used to consider for likelihood.
    1. The longer the duration between enrolment date and registration date,
    the more likely the parent will take up; AND
    2. The longer the duration between today and enrolment date, the less likely
    the parent will take up; AND
    3. The higher the rank, the higher the likelihood

    :param enrolment_date: Date the parent would like to enrol the child
    :param reg_date: Date the parent register the interest
    :param rank: Ranking of parent on this childcare
    :return: Likelihood category
    """
    # If parent looking to enrol before today or has registered after today, data not valid
    if enrolment_date < TODAY or reg_date > TODAY:
        return "N.A."
    # If parent register to enrol earlier e.g., register on Feb 2020 for Jan 2020 vacancy
    elif reg_date > enrolment_date:
        return "N.A."
    elif not 1 <= rank <= 10:
        return "N.A."
    else:
        enrol_reg_diff = (enrolment_date.year - reg_date.year) * 12 + (enrolment_date.month - reg_date.month)
        today_reg_diff = (TODAY.year - reg_date.year) * 12 + (TODAY.month - reg_date.month)
        enrol_reg_prob = 1 if enrol_reg_diff*(1/11) > 1 else enrol_reg_diff*(1/11)
        today_reg_prob = 1 - today_reg_diff*(1/11)
        rank_prob = 1 - ((rank - 1)/10)
        final_probability = round(min(enrol_reg_prob, today_reg_prob, rank_prob), 2)
        likelihood = ""
        if final_probability <= 0.3:
            likelihood = "Low"
        elif 0.3 < final_probability <= 0.8:
            likelihood = "Medium"
        else:
            likelihood = "High"

        return likelihood


def batch_predict_likelihood(centre_code):
    """
    - First portion is just to grab all childcare data from sql and add columns like enrol_reg_diff
    (Difference between enrolment and registration date) and today_reg_diff (Diff between today and
    registration date) which are required for prediction
    - Second portion is to get GA chromosomes that are proven to yield High likelihood, then we compare
    the childcare data against the chromosomes to quickly find data that will yield High likelihood
    - If there are leftover childcare data that does not match any chromosomes, then we will predict
    them here.
    - Lastly, update the predicted likelihoods to the database.
    :param centre_code: Centre code in string
    :return: None
    """
    choices_df = pd.read_sql(f"SELECT child_idno, cc_action, childcare_rank, reg_date, parent_choice_key "
                             f"FROM parent_choices "
                             f"WHERE centre_code='{centre_code}'", con=db_connection)
    past_choices_df = choices_df[choices_df["cc_action"].notnull()]

    if len(past_choices_df.index) < 50:
        return

    formatted_child_idnos = "','".join(choices_df["child_idno"].to_list())
    choices_df["cc_action"] = choices_df["cc_action"].map(lambda x: x.replace("\r", "")
                                                          if x is not None
                                                          else x)
    details_df = pd.read_sql(f"SELECT child_birthdate, enrolment_date, study_level, "
                             f"acceptable_distance, acceptable_fees, second_language, "
                             f"dietary_restrictions, service_type, child_idno "
                             f"FROM parent_details "
                             f"WHERE child_idno IN ('{formatted_child_idnos}')", con=db_connection)
    details_df["service_type"] = details_df["service_type"].map(lambda x: x.replace("\r", "")
                                                                if x is not None
                                                                else x)
    full_details = details_df.merge(choices_df,
                                    how="inner",
                                    on="child_idno")
    full_details["enrol_reg_diff"] = full_details.apply(lambda x: ((x["enrolment_date"].year - x["reg_date"].year) * 12
                                                                   + (x["enrolment_date"].month - x["reg_date"].month)),
                                                        axis=1)
    full_details["enrol_reg_diff"] = full_details["enrol_reg_diff"].map(lambda x: x
                                                                        if x >= 0
                                                                        else 0)
    full_details["today_reg_diff"] = full_details.apply(lambda x: ((TODAY.year - x["reg_date"].year) * 12
                                                                   + (TODAY.month - x["reg_date"].month)),
                                                        axis=1)
    full_details["today_reg_diff"] = full_details["today_reg_diff"].map(lambda x: x
                                                                        if x >= 0
                                                                        else 0)
    full_details = full_details.drop(["child_birthdate", "enrolment_date", "reg_date"], axis=1)
    details_for_dt = full_details.drop(["child_idno", "parent_choice_key"], axis=1)

    pipeline, best_model = get_dt_model(details_for_dt)
    rules_var_values = get_top_ga_solutions(pipeline, details_for_dt, best_model)
    predicted_df = add_likelihood_from_dt(details_for_dt, pipeline, best_model, rules_var_values)

    full_details["enrol_reg_diff"] = full_details["enrol_reg_diff"].astype(str)
    full_details["today_reg_diff"] = full_details["today_reg_diff"].astype(str)
    full_details["childcare_rank"] = full_details["childcare_rank"].astype(str)

    full_details_w_predictions = full_details.merge(predicted_df,
                                 how="inner",
                                 on=["study_level", "acceptable_distance", "acceptable_fees",
                                     "second_language", "dietary_restrictions", "service_type", "childcare_rank",
                                     "enrol_reg_diff", "today_reg_diff"])
    final_df = full_details_w_predictions[["parent_choice_key", "predictions"]]

    cur = mysql.connection.cursor()
    for _, row in final_df.iterrows():
        cur.execute(f"UPDATE parent_choices "
                    f"SET likelihood = '{row['predictions']}' "
                    f"WHERE parent_choice_key={row['parent_choice_key']};")
        mysql.connection.commit()


if __name__ == '__main__':
    # batch_predict_likelihood("PT9179")
    app.run(debug=True)
    # print(get_probability(datetime(2021, 9, 1), datetime(2020, 6, 1), 3))
    # batch_predict_likelihood()
