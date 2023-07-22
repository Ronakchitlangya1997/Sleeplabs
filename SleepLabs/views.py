from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from pytz import timezone
from SleepLabs.models import SleepLab, SleepLabOptv1
import json
import datetime
import numpy as np
import math
from django.contrib.auth.decorators import login_required 

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import Span

from django.http import JsonResponse

import pandas as pd
import math
from django.contrib.auth import authenticate, login as UserLogin, logout as UserLogout, update_session_auth_hash
from django.contrib.auth.models import User
EXCEL = 1

@login_required(login_url='login')
def home(request):
    return render(request, 'sleeplabs.html')


@csrf_exempt
def deviceData(request):
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")
    print("Current Time : ",dt_string)


    if request.method =="POST":
        bodyDecoded = request.body.decode('utf-8')
        bodyJson = json.loads(bodyDecoded)

        deviceID = bodyJson['DeviceID']
        print('Received data from ' + deviceID)

        time = now
        dataSample = SleepLabOptv1(DevID=deviceID, timestamp=time,jsonData=bodyJson)
        dataSample.save()

        return HttpResponse('Device Data Saved')
    
    return HttpResponse('Request Method Error')

def converStringToDict(d) :

    d = d.replace("\'", "\"")
    d = json.loads(d)

    return d

def dict_len(d):

    return len(d)

# Function to convert dictionary to Pandas Series
def dict_to_series(d):
    return pd.Series(d)

# Apply function to each row of the column containing dictionaries
def timestampKey(t, d, id):

    time = t

    for i in range(len(d) - 1):

        sample = 'S' + str(len(d) - 1 - i -1)
        #print(sample)

        packetTime = time
        
        d[sample]['time'] = packetTime

        time = time - datetime.timedelta(milliseconds=250)

    return d

#Helper Functions
def addMagnitude(df):
    sample_rate = 250
    df['AcX'] = df['AcX'].astype('int')
    df['AcY'] = df['AcY'].astype('int')
    df['AcZ'] = df['AcZ'].astype('int')
    #df = df[::-1].reset_index(drop=True)
    df['Magnitude'] = np.sqrt(df['AcX']**2 + df['AcY']**2 + df['AcZ']**2) / 2048
    df['Timestamp'] = pd.to_datetime(df['time'])

    return df

def bedOccData (df, occToggle, occThresh) :
    df['OcV'] = df['OcV'].astype(float)

    # Bed Occupancy Calculations
    sample_rate = 250
    
    if (occToggle == 1) :
        df = df[df['OcV'].astype(float) > occThresh].reset_index()
    else:
        pass

    time = df['Timestamp'].values
    
    
    try :
        sample_rate = int(np.round((time[1] - time[0]) / np.timedelta64(1, 'ms')))
        bedOccupantTime = int(np.round((time[-1] - time[0]) / np.timedelta64(1, 'ms')))
    except :
        print("No Occupant Activity")
        bedOccupantTime = 0

    return df, abs(sample_rate)

def magArrayProcessing (array, df) :
    sample_rate = 250
    
    # Total 
    totalPoints = len(np.where(array < 100)[0]) * sample_rate // 1000
    
    # Calculate the average of the array
    avg = np.mean(array)

    print("Average :", avg)

    # Set a threshold as the average
    thresholdPositive = avg+0.03
    thresholdNegative = avg-0.03

    # Count the number of values greater than the threshold
    countPositive = np.sum(array > thresholdPositive)
    countNegative = np.sum(array < thresholdNegative)

    print("Points :")
    print(countPositive)
    print(countNegative)
    
    totalMovingPoints = countPositive + countNegative
    
    totalMovingPoints = totalMovingPoints*sample_rate // 1000

    # Create 'Movement' column
    df['Movement'] = "0"

    # Apply condition to set 'Orientation' to 'Vertical'

    df.loc[(abs(df['Magnitude']) > thresholdPositive) | (abs(df['Magnitude']) < thresholdNegative), 'Movement'] = "1"


    if EXCEL :
        df.to_excel('./SleepLabs/Data/movementDistribution.xlsx')

    
    return totalPoints, countNegative, countPositive, totalMovingPoints

def secondstoHours(secs):
    sample_rate = 250
    kpi = []
    for sec in secs :
        sec = int(sec)
        kpi.append(str(datetime.timedelta(seconds=sec)))
    return kpi

def computeDistribution(df) :
    sample_rate = 250
    #magBins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.94, 0.96, 0.97,1, 1.2, 1.23, 1.26, 1.29, 1.3, 1.4, 1.5, float('inf')]
    magBins = [0, 0.5, 1, 1.5, 2,2.5, 3, 3.5, float('inf')]
    magCounts = pd.cut(df['Magnitude'], bins=magBins).value_counts()
    
    print("Magnitude Distribution")
    print(magCounts.sort_index())
    print(type(magCounts.sort_index()))
    magCounts.sort_index().to_excel('./SleepLabs/Data/magnitudeDistribution.xlsx')


    # Orientation

    # Convert columns to float data type
    df[['AcX', 'AcY', 'AcZ']] = df[['AcX', 'AcY', 'AcZ']].astype(float)

    # Create 'Orientation' column
    df['Orientation'] = "Horizontal"

    # Apply condition to set 'Orientation' to 'Vertical'
    df.loc[(abs(df['AcX']) > abs(df['AcZ'])) | (abs(df['AcY']) > abs(df['AcZ'])), 'Orientation'] = "Vertical"



    # df['Orientation'] = "Horizontal"
    # df.loc[(abs(int(df['AcX'])) > abs(int(df['AcZ']))) or (abs(int(df['AcY']))>abs(int(df['AcZ']))) , 'Orientation'] = "Vertical"
    
    if EXCEL :
        df.to_excel('./SleepLabs/Data/orientation.xlsx')

    occVBins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, float('inf')]
    occVCounts = pd.cut(df['OcV'], bins=occVBins).value_counts()
    
    print("occV Distribution")
    print(occVCounts.sort_index())    

    occVCounts.sort_index().to_excel('./SleepLabs/Data/voltageDistribution.xlsx')

def totalTime (df) :
    sample_rate = 250
    OcV = (df['OcV'].astype(float)).values
    totalPoints = len(np.where(OcV < 10000)[0]) * sample_rate // 1000
    totalTimeHR = secondstoHours([totalPoints])[0]
    
    return totalTimeHR, totalPoints
    
def getSleepRating(sleep_time, awake_time,rationIndexes):

    try :
        total_time = sleep_time + awake_time
        sleep_ratio = round(sleep_time / total_time,2)
        
        if sleep_ratio >= rationIndexes[0]:
            rating = 5
            quality = "Excellent"
        elif sleep_ratio >= rationIndexes[1]:
            rating = 4
            quality = "Good"
        elif sleep_ratio >= rationIndexes[2]:
            rating = 3
            quality = "Fair"
        elif sleep_ratio >= rationIndexes[3]:
            rating = 2
            quality = "Poor"
        else:
            rating = 1
            quality = "Very poor"
    except :
        rating = -1
        sleep_ratio = -1
        quality = "NA"
        
    return sleep_ratio, rating, quality


@csrf_exempt
def sleep_labs_graph_api_v3(request):

    
    #Fetch All
    #df = pd.DataFrame.from_records(SleepLabOptv1.objects.all().values())

    #Fetch certain date
    #date_str = '2023-03-29'

    date_str_start = '2023-03-31'
    date_str_end = '2023-03-31'
    start_time_str = '00:00:00'
    end_time_str = '08:00:00'

    #Fetch from a certain timestamp

    start_datetime_str = date_str_end + ' ' + start_time_str
    end_datetime_str = date_str_end + ' ' + end_time_str

    if request.method == "POST":
        sleep_data = {}
        jsondata = json.loads(request.body)
        print("Printing JSON ")
        print(jsondata)
        date_str_start = jsondata['Date']
        date_str_end = jsondata['Date']
        devEUI = jsondata['devEUI']
        start_time_str = jsondata['StartTimehours']+":"+jsondata['StartTimemin']+":"+jsondata['StartTimesec']
        end_time_str = jsondata['EndTimehours']+":"+jsondata['EndTimemin']+":"+jsondata['EndTimesec']

        print(date_str_start, date_str_end, start_time_str, end_time_str)

        #Dummy
        # date_str_start = '2023-05-04'
        # date_str_end = '2023-05-05'
        # start_time_str = '22:00:00'
        # end_time_str = '10:00:00'

        start_datetime_str = date_str_start + ' ' + start_time_str
        end_datetime_str = date_str_end + ' ' + end_time_str

        
        df = pd.DataFrame.from_records(SleepLabOptv1.objects.filter(DevID =devEUI, timestamp__gte=start_datetime_str, timestamp__lte=end_datetime_str).values())
        

        print("Fetching the raw dataframe :")
        print(df)
        print(df.info())

        print("Missing dB")
        df_2 = df.set_index('timestamp')
        
        df_2.index = pd.to_datetime(df_2.index)

        hourly_counts = df_2.resample('H').count()
        print(hourly_counts)
        print(type(hourly_counts))
        hourly_counts.to_excel('./SleepLabs/Data/rawPSRDistribution.xlsx')
        if EXCEL :
            df.to_excel('./SleepLabs/Data/rawData.xlsx')

        # ***************************** DUMMY ****************************** #

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # set timestamp column as index
        df.set_index('timestamp', inplace=True)

        # resample to one minute interval and fill missing values with the previous row
        df_resampled = df.resample('1T').ffill()

        # reset index
        df_resampled.reset_index(inplace=True)

        df = df_resampled

        df = df.drop(0)

        print(df)
        df_2 = df.set_index('timestamp')
        df_2.index = pd.to_datetime(df_2.index)

        hourly_counts = df_2.resample('H').count()
        print(hourly_counts)
        print(type(hourly_counts))

        hourly_counts.to_excel('./SleepLabs/Data/extrapolatedPSRDistribution.xlsx')
        #df.to_excel('./SleepLabs/Data/extrapolatedData.xlsx')



        #Dummy

        # df = pd.read_excel('./rawDataDummyFinal.xlsx')
        # print(df)
        # df = df.drop(0)
        # print(df.info())
        
        algo_Data = processSleepData(df)
        print(algo_Data)

        request.session['algo_Data'] = algo_Data
        return HttpResponse(algo_Data)
    

def processSleepData(df) :
    #df = pd.read_csv('./rawData.csv')

    df = pd.DataFrame(df)
    print(df)
    print(df.dtypes)

    #df['jsonData'] = df['jsonData'].apply(converStringToDict)

    df['dict_len'] = df['jsonData'].apply(dict_len)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['jsonData'] = df.apply(lambda row: timestampKey(row['timestamp'], row['jsonData'], row['auto_increment_id']), axis=1)

    df_new = pd.DataFrame(columns=['AcX', 'AcY', 'AcZ', 'GyX', 'GyY', 'GyZ', 'Oc2', 'OcV', 'Occ', 'OcV2', 'time', 'id'])

    missing_samples = []  # list to store missing samples for each row

    # loop over each row
    for index, row in df.iterrows():
        d= row['jsonData']
        
        try :
            del d['DeviceID']
        except :
            pass

        x = []
        for i in range(len(d) - 1):
            sample = 'S' + str(len(d) - 1 - i -1)
            if sample not in d:
                missing_samples.append(sample)  # add missing sample to list
                continue
            d[sample]['id'] = sample
            x.append(d[sample])

        df = pd.DataFrame(x)
        frames = [df_new, df]
        df_new = pd.concat(frames)
    print("Missing Samples")
    print(missing_samples)  # print list of missing samples for all rows

    print(df_new)

    jsonapidata = algov2(df_new)
    #df_new.to_csv('./processedData.csv')

    return jsonapidata


def algov2(df) :

    #df = pd.read_csv('./processedData.csv')
    #df.to_csv('./procssedSleepData.csv')


    print("Running the algorithm ...............")


    occToggle = 1
    occThresh = 200

    sleep_data = {}

    sample_rate = 250

    df = df[::-1]

    totalTimeHR, totalPoints = totalTime(df)

    print("Adding Magnitude...")
    df = addMagnitude(df)


    print("Computing Bed Occupancy...")
    df, sample_rate = bedOccData(df, occToggle, occThresh)


    print("Hour KPIs Computation...")
    totalOccPoints, countNegative, countPositive, totalMovingPoints = magArrayProcessing(df['Magnitude'].values, df)

    print("Computing Distribution...")
    computeDistribution(df)

    print("Converting to proper hours...")
    kpi = secondstoHours([totalOccPoints, totalMovingPoints, totalOccPoints-totalMovingPoints])

    print("Results...")
    print("Total Time :", totalTimeHR)
    print("Occupant Time : ", kpi[0])
    print("Movement Time : ", kpi[1])

    sleep_data['total_time'] =  totalTimeHR
    sleep_data['sample_rate'] =  sample_rate
    sleep_data['bedOccupantTime'] =  kpi[0]
    sleep_data['awake_time'] =  kpi[1]
    sleep_data['sleep_time'] =  kpi[2]
    

    

    if sleep_data['bedOccupantTime'] == "00:00:00" :
        print("In the if statement of Zero Checks")
        print("Bed Occupant Time is", sleep_data['bedOccupantTime'])
        Sleep_info = "Summary Not Available"
    
    else :
        Sleep_info = "Of the" + " " + str(sleep_data['total_time']) + " " "of monitoring :" + "\n" + "Your pillow was occupied for" + " " + str(int((totalOccPoints/totalPoints)*100)) + "%" + " " + "of the time." + "\n" + "Your pillow was unoccupied for" + " " + str(int(100- int((totalOccPoints/totalPoints)*100))) +"%"+ " of the time." + "\n" + "During the bed occupant hours, you slept restfully for" + " " + str( int(((totalOccPoints-totalMovingPoints)/(totalMovingPoints+(totalOccPoints-totalMovingPoints)))*100) ) + "%" + " of the entire sleep duration."
    
    sleep_data['Sleep_info'] = Sleep_info

    print(Sleep_info)
    # Get from front-end otherwise default as following :
    rationIndexes = [0.95, 0.9, 0.8, 0.7]

    Sleep_ratio, Sleep_star, Sleep_quality = getSleepRating(totalOccPoints-totalMovingPoints, totalMovingPoints,rationIndexes)
    
    print("Sleep Quality Outputs :")
    print("Sleep Ratio :", Sleep_ratio)
    print("Sleep Star :", Sleep_star)
    print("Sleep Quality :", Sleep_quality)
    
    sleep_data['Sleep_ratio'] = Sleep_ratio
    sleep_data['Sleep_star'] = Sleep_star
    sleep_data['Sleep_quality'] = Sleep_quality

    print("Analysis Complete ...................................")
    # Find the average magnitude of all rows between 0.7 and 1.0
    # avg = df.loc[df['Magnitude'].between(0.7, 1.0), 'Magnitude'].mean()

    # # Highlight rows with magnitude outside the range of (avg - 0.3) to (avg + 0.3)
    # highlight_mask = (df["Magnitude"] < avg - 0.3) | (df["Magnitude"] > avg + 0.3)

    # print(highlight_mask)

    


    # df.loc[(df['Magnitude'] <= avg - 0.3) or (df['Magnitude'] >= avg + 0.3), 'highlight_mask'] = 1
    # df.loc[(df['Magnitude'] >= avg - 0.3) or (df['Magnitude'] <= avg + 0.3), 'highlight_mask'] = 0
    if EXCEL :
        df.to_csv('./SleepLabs/Data/withmag.csv')


    #highlighted_rows = ColumnDataSource(df.loc[highlight_mask])
    

    jsonapidata = json.dumps(sleep_data)

    return jsonapidata




def orientation(request) :

    devID = 'F0230003'

    obj= SleepLabOptv1.objects.filter(DevID=devID).order_by('-auto_increment_id')[0]

    data = obj.jsonData

    print(type(obj.jsonData))

    print(data['S239'])

    AcX = data['S239']['AcX']
    AcY = data['S239']['AcY']
    AcZ = data['S239']['AcZ']

    if(int(AcX) > int(AcZ) or int(AcY)>int(AcZ)) :
        orientation = 'Vertical'
    else :
        orientation = 'Flat'
    
    print("Orientation : ", orientation)

    

    return HttpResponse('ok')

@csrf_exempt
def Devicestatus(request):

    print("In Device Status")

    if request.method == "POST":
        sleep_data = {}
        jsondata = json.loads(request.body)
        print(jsondata)
        latest_datetime = SleepLabOptv1.objects.filter(DevID=jsondata).order_by('-timestamp').first()
        print(latest_datetime)
        previous_datetime= datetime.datetime.strptime(latest_datetime.timestamp, '%Y-%m-%d %H:%M:%S.%f') 
        print('previous_datetime', previous_datetime)
        present_datetime = datetime.datetime.now()
        # Calculate the difference between the two dates
        time_difference = present_datetime - previous_datetime


        data = {}

        # Check if the difference is greater than 1 minute
        if time_difference < datetime.timedelta(minutes=2):
            data = {
                'Device': 'Active',
            }
        else:
             data = {
                'Device': 'Inactive',
            }
        output_datetime_str = previous_datetime.strftime("%d %b, %I:%M %p")
        data['lastActive'] = output_datetime_str
        print(data['lastActive'])

        obj= SleepLabOptv1.objects.filter(DevID=jsondata).order_by('-auto_increment_id')[0]

        obj_data = obj.jsonData

        print(type(obj.jsonData))

        print(obj_data['S239'])

        AcX = obj_data['S239']['AcX']
        AcY = obj_data['S239']['AcY']
        AcZ = obj_data['S239']['AcZ']
        OcV = obj_data['S239']['OcV']
        # Occupancy Code
        if int(OcV) > 200 :
            data['occupancy'] = 'Occupied'
        else :
            data['occupancy'] = 'Unoccupied'

        if(abs(int(AcX)) > abs(int(AcZ)) or abs(int(AcY))>abs(int(AcZ))) :
            orientation = 'Vertical'
            data['orientation'] = 'Vertical'
        else :
            orientation = 'Horizontal'
            data['orientation'] = 'Horizontal'
        
    return JsonResponse(data)




@csrf_exempt
def deviceStatusMobileApp(request):

    if request.method == "GET":

        jsondata = json.loads(request.body)
        jsondata = jsondata['DeviceID']
        print("Device ID", jsondata)

        latest_datetime = SleepLabOptv1.objects.filter(DevID=jsondata).order_by('-timestamp').first()
        print('Latest Datetime', latest_datetime)
        previous_datetime= datetime.datetime.strptime(latest_datetime.timestamp, '%Y-%m-%d %H:%M:%S.%f') 
        print('Previous Datetime', previous_datetime)
        present_datetime = datetime.datetime.now()

        # Calculate the difference between the two dates
        time_difference = present_datetime - previous_datetime


        data = {}

        # Check if the difference is greater than 1 minute
        if time_difference < datetime.timedelta(minutes=2):
            data = {
                'status': 'Active',
            }
        else:
             data = {
                'status': 'Inactive',
            }
        output_datetime_str = previous_datetime.strftime("%d %b, %I:%M %p")
        data['lastActive'] = output_datetime_str
        print('Last Active Timestamp', data['lastActive'])

        obj= SleepLabOptv1.objects.filter(DevID=jsondata).order_by('-auto_increment_id')[0]

        obj_data = obj.jsonData


        AcX = obj_data['S239']['AcX']
        AcY = obj_data['S239']['AcY']
        AcZ = obj_data['S239']['AcZ']
        OcV = obj_data['S239']['OcV']
        # Occupancy Code
        if int(OcV) > 200 :
            data['occupancy'] = 'Occupied'
        else :
            data['occupancy'] = 'Unoccupied'

        if(abs(int(AcX)) > abs(int(AcZ)) or abs(int(AcY))>abs(int(AcZ))) :
            orientation = 'Vertical'
            data['orientation'] = 'Vertical'
        else :
            orientation = 'Horizontal'
            data['orientation'] = 'Horizontal'
    
    print("Device Status Response :", data)

    return JsonResponse(data)







def sleephighligh(request):
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, DataTable, TableColumn
    from bokeh.layouts import column
    import pandas as pd

    data = pd.read_csv('withmag.csv')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['highlight_mask'].astype(int)
    df = data
    print(df)

    source = ColumnDataSource(df)
    
    p = figure(plot_width=1320, plot_height=650,title="Magnitude vs Timestamp", x_axis_type='datetime', x_axis_label='Timestamp', y_axis_label='Magnitude')

    # Add the line chart
    p.line(x='Timestamp', y='Magnitude', source=source, line_width=2)

    # Add a circle glyph for the points where avg == 1
    p.circle(x='Timestamp', y='Magnitude', source=df[df['highlight_mask'] == 1], fill_color='red', size=8)

    script, div = components(p)

    return render(request, 'sleeplabsgraph.html', {'script': script, 'div': div})

def Sleeplabsgraph(request):
    # Define parameters
    #create a plot 
    from bokeh.io import output_notebook, show
    from bokeh.plotting import figure
    from bokeh.models import Span
    result = request.session["algo_Data"]
    print('My session: ', result)
    # Define parameters
    total_time = 8*60*60     # Total time in seconds
    sample_rate = 150        # Sample rate in milliseconds
    num_samples = int(total_time*1000/sample_rate)
    sleep_time = 6*60*60     # Sleep time in seconds
    awake_time = total_time - sleep_time
    move_duration = 30       # Movement duration in seconds
    move_freq = 30            # Movement frequency per hour
    move_periods = int((awake_time/3600) * move_freq)
    move_timestamps = np.sort(np.random.choice(range(sleep_time), size=move_periods, replace=False))

    # Generate data
    time = np.arange(num_samples) * sample_rate / 1000.0
    magnitude = np.ones(num_samples)
    for ts in move_timestamps:
        move_start = int((ts/sample_rate) * 1000)
        move_end = int(((ts+move_duration)/sample_rate) * 1000)
        magnitude[move_start:move_end] = np.random.uniform(1.2, 2.5)

    # Create dataframe
    data = {'Time': time, 'Magnitude': magnitude}
    df = pd.DataFrame(data)

    # Output to Jupyter notebook
    output_notebook()

    # Create a figure
    p = figure(plot_width=1320, plot_height=650, x_axis_label='Time (s)', y_axis_label='Acceleration Magnitude', title='Sleep Data')

    # Add a line glyph
    p.line(df['Time'], df['Magnitude'], line_width=2, color='darkgreen')

    # Highlight movement periods
    for ts in move_timestamps:
        start_time = ts - move_duration/2
        end_time = ts + move_duration/2
        span = Span(location=ts, dimension='height', line_color='red', line_alpha=0.2)
        p.add_layout(span)
    
    # Show the figure
    # show(p)
 
    script, div = components(p)

    return render(request, 'sleeplabsgraph.html', {'script': script, 'div': div})

