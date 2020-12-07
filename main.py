from flask import Flask
from flask import request, jsonify, render_template
import xgboost
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

CrimeID = {
            'abuse':0,
            'accident':1,
            'assault':2,
            'burglary':3,
            'drugs':4,
            'felony':5,
            'kidnapping':6,
            'other':7,
            'sex related':8,
            'vandalism':9,
            'weapon violation':10
          }

YearID = {'2014':0,'2015':1}

MonthID = {'jan':0,'feb':1,'mar':2,'apr':3,'may':4,'jun':5,'july':6,'aug':7,'sep':8,'oct':9,'nov':10,'dec':11}

DayID = {'mon':0,'tue':1,'wed':2,'thu':3,'fri':4,'sat':5,'sun':6}

TimeID = {'morning':0,'afternoon':1,'evening':2,'night':3}

CityID = {'aus':0,'balt':1,'bos':2,'chic':3,'den':4,'la':5,'nyc':6}

def piechart(data):
    x = np.char.array(
        ['Morning(5:00am-11:00am)', 'afternoon(11:00am-5:00pm)', 'Evening(5:00pm-10:00pm)', 'Night(10:00pm-5:00am)'])
    y = np.array(data)
    colors = ['#2980B9', '#76D7C4', '#16A085', '#85C1E9']
    porcent = 100. * y / y.sum()
    plt.figure(figsize=(10, 5))
    patches, texts = plt.pie(y, colors=colors, radius=1.2)
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x, porcent)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, y),
                                             key=lambda x: x[2],
                                             reverse=True))
    plt.title("Crime prediction by time")
    plt.legend(patches, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=10,
               bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.96)
    plt.savefig('./static/piechart.png', bbox_inches="tight")  # ,bbox_inches="tight"
    # plt.show()

def barchart(value):
    values = value


    label = ['abuse', 'accident', 'assault', 'burglary', 'drugs', 'felony', 'kidnapping', 'other', 'sex related',
             'vandalism', 'weapon violation']

    Z = [x for _, x in sorted(zip(values, label))]
    values.sort()
    colour = ['#9ad3bc', '#9ad3bc', '#9ad3bc', '#9ad3bc', '#16a596', '#16a596', '#16a596', '#16a596', '#aa3a3a',
              '#aa3a3a', '#aa3a3a', ]

    ypos = np.arange(len(Z))
    plt.title("PREDICT BY TYPE OF CRIME")
    # plt.tight_layout()
    plt.ylabel("types of crimes")
    plt.xlabel("probabilty")
    plt.yticks(ypos, Z)
    plt.barh(ypos, values, color=colour)
    # plt.subplots_adjust(left=0.1, bottom=0.0, right=0.91, top=0.91)
    plt.tight_layout()
    plt.savefig('./static/barchart.png')
    # plt.show()

with open('model/CityPredict.pkl', 'rb') as f_in1:
    cityModel = pickle.load(f_in1)

with open('model/Crime.pkl', 'rb') as f_in2:
    crimeModel = pickle.load(f_in2)

with open('model/TimeCategory.pkl', 'rb') as f_in3:
    timeModel = pickle.load(f_in3)

app = Flask(__name__)
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/')
def hello_world():
   return render_template('index.html')

@app.route('/predictBycrime')
def renderCrimePage():
    return render_template('predictByCrime.html')

@app.route('/predictBycity')
def renderCityPage():
    return render_template('predictByCity.html')

@app.route('/predictBytime')
def renderTimePage():
    return render_template('predictByTime.html')


@app.route('/predictCrimeType',methods=['POST'])
def predictCrimeType():
    city = request.form['city']
    time = request.form['time']
    lat = request.form['latitude']
    lon = request.form['longitude']
    month = request.form['month']
    year = request.form['year']
    day = request.form['day']

    data = []
    data.append(YearID[year])
    data.append(MonthID[month])
    data.append(DayID[day])
    data.append(TimeID[time])
    data.append(CityID[city])
    data.append(int(lat))
    data.append(int(lon))
    res = crimeModel.predict_proba(np.array([data]))
    print([data])
    print(res)
    barchart(res[0].tolist())
    return render_template('predictByCrime.html')


@app.route('/predictTimeCategory',methods=['POST'])
def predictTimeCategory():
    city = request.form['city']
    lat = request.form['latitude']
    lon = request.form['longitude']
    month = request.form['month']
    year = request.form['year']
    crime = request.form['crime']
    day = request.form['day']

    data = []
    data.append(CrimeID[crime])
    data.append(YearID[year])
    data.append(MonthID[month])
    data.append(DayID[day])
    data.append(CityID[city])
    data.append(int(lat))
    data.append(int(lon))
    res = timeModel.predict_proba(np.array([data]))
    print([data])
    print(res[0])
    piechart(res[0].tolist())
    return render_template('predictByTime.html')

@app.route('/predictCity',methods=['POST'])
def predictCity():
    month = request.form['month']
    year = request.form['year']
    crime = request.form['crime']
    time = request.form['time']
    day = request.form['day']

    data = []
    data.append(CrimeID[crime])
    data.append(YearID[year])
    data.append(MonthID[month])
    data.append(DayID[day])
    data.append(TimeID[time])
    res = cityModel.predict_proba(np.array([data]))
    print([data])
    print(res[0])
    return render_template('predictByCity.html')


if __name__ == '__main__':
   app.run(debug=True)