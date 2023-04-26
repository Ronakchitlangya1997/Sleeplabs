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

# Create your views here.

@login_required(login_url='login')
def home(request):
    return render(request, 'sleeplabs.html')

@csrf_exempt
def SleeplabsAPI(request):
    if request.method =="POST":
        print(request.body)
        # data = b'AcX = 427, AcY = 1550, AcZ = -1572, GyX = -711, GyY = 1895, GyZ = -156 \r\n'
        post_data = request.body.decode()
        data = post_data.split(",")
        split_data = []
        for ele in data:
            split_data.append(ele.split('='))
        print(split_data)
        Sleep_Labsobject=SleepLab(AcX=split_data[0][1], AcY=split_data[1][1], AcZ=split_data[2][1],
                                        GyX=split_data[3][1], GyY=split_data[4][1], GyZ=split_data[5][1], OCC=split_data[6][1])
        Sleep_Labsobject.save()
        return HttpResponse('ok')
    return HttpResponse('Not working')

import pandas as pd
import math

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

        packetTime = time
        
        d[sample]['time'] = packetTime

        time = time - datetime.timedelta(milliseconds=250)

    return d

def sleep_labs_graph_api_v2(request):

    
    #Fetch All
    #df = pd.DataFrame.from_records(SleepLabOptv1.objects.all().values())

    #Fetch certain date
    #date_str = '2023-03-29'
    #df = pd.DataFrame.from_records(SleepLabOptv1.objects.filter(timestamp__startswith=date_str).values())

    #Fetch from a certain timestamp

    if request.method == "POST":

        
        sleep_data = {}
        jsondata = json.loads(request.body)
        date_str_start = jsondata['Date']
        date_str_end = jsondata['Date']
        start_time_str = jsondata['StartTimehours']+":"+jsondata['StartTimemin']+":"+jsondata['StartTimesec']
        end_time_str = jsondata['EndTimehours']+":"+jsondata['EndTimemin']+":"+jsondata['EndTimesec']

        print(date_str_start, date_str_end, start_time_str, end_time_str)
        # date_str_start = '2023-03-29'
        # date_str_end = '2023-03-29'
        # start_time_str = '16:00:00'
        # end_time_str = '17:00:00'

        start_datetime_str = date_str_end + ' ' + start_time_str
        end_datetime_str = date_str_end + ' ' + end_time_str

        df = pd.DataFrame.from_records(SleepLabOptv1.objects.filter(timestamp__gte=start_datetime_str, timestamp__lte=end_datetime_str).values())

        print("Fetching the raw dataframe :")
        print(df)
        

        df.to_csv('./rawData.csv')
        processSleep_Data = processSleepData()

        algo_Data = algov2()
        
        return HttpResponse(algo_Data)
   
    return HttpResponse('ok')


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
        date_str_start = jsondata['Date']
        date_str_end = jsondata['Date']
        start_time_str = jsondata['StartTimehours']+":"+jsondata['StartTimemin']+":"+jsondata['StartTimesec']
        end_time_str = jsondata['EndTimehours']+":"+jsondata['EndTimemin']+":"+jsondata['EndTimesec']

        print(date_str_start, date_str_end, start_time_str, end_time_str)
        # date_str_start = '2023-03-29'
        # date_str_end = '2023-03-29'
        # start_time_str = '16:00:00'
        # end_time_str = '17:00:00'

        start_datetime_str = date_str_end + ' ' + start_time_str
        end_datetime_str = date_str_end + ' ' + end_time_str

        df = pd.DataFrame.from_records(SleepLabOptv1.objects.filter(timestamp__gte=start_datetime_str, timestamp__lte=end_datetime_str).values())

        print("Fetching the raw dataframe :")
        print(df)

        print("Missing dB")
        df_2 = df.set_index('timestamp')
        df_2.index = pd.to_datetime(df_2.index)

        hourly_counts = df_2.resample('H').count()
        print(hourly_counts)
        # df.to_excel('./rawData.xlsx')
        
        algo_Data = processSleepData(df)
        print(algo_Data)

        request.session['algo_Data'] = algo_Data
        return HttpResponse(algo_Data)
    
   


# def processSleepData(df) :
#     #df = pd.read_csv('./rawData.csv')

#     df = pd.DataFrame(df)
#     print(df)
#     print(df.dtypes)

#     #df['jsonData'] = df['jsonData'].apply(converStringToDict)

#     df['dict_len'] = df['jsonData'].apply(dict_len)

#     df['timestamp'] = pd.to_datetime(df['timestamp'])

#     df['jsonData'] = df.apply(lambda row: timestampKey(row['timestamp'], row['jsonData'], row['auto_increment_id']), axis=1)

#     df_new = pd.DataFrame(columns=['AcX', 'AcY', 'AcZ', 'GyX', 'GyY', 'GyZ', 'Oc2', 'OcV', 'Occ', 'OcV2', 'time', 'id'])


#     y = []
#     # loop over each row
#     for index, row in df.iterrows():
#         d= row['jsonData']
#         del d['DeviceID']

#         x = []
#         for i in range(len(d) - 1):
#             sample = 'S' + str(len(d) - 1 - i -1)
#             d[sample]['id'] = sample
#             x.append(d[sample])

#         df = pd.DataFrame(x)
#         frames = [df_new, df]
#         df_new = pd.concat(frames)

#     print(df_new)

#     jsonapidata = algo(df_new)
#     #df_new.to_csv('./processedData.csv')

#     return jsonapidata


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
        del d['DeviceID']

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

# def algo(df) :

#     #df = pd.read_csv('./processedData.csv')


#     print("Running the algorithm .............................")

#     # From Frontend 
#     bedOccToggleOne = 0
#     bedOccToggleTwo = 0
#     bedOccThresholdOne = 1.1
#     bedOccThresholdTwo = 1.1

#     print("Bed Occupancy Inputs :")
#     print(f"Sensor-1 :: Toggle ; Threshold : {bedOccToggleOne} ; {bedOccThresholdOne}")
#     print(f"Sensor-2 :: Toggle ; Threshold : {bedOccToggleTwo} ; {bedOccThresholdTwo}")


#     sleep_data = {}

#     df = df[::-1]
#     x = []
#     df['AcX'] = df['AcX'].astype('int')
#     df['AcY'] = df['AcY'].astype('int')
#     df['AcZ'] = df['AcZ'].astype('int')

#     for index, row in df.iterrows():
#         x.append(math.sqrt(row['AcX']**2 + row['AcY']**2 + row['AcY']**2))
        
#     df['Magnitude'] = x
#     df['Magnitude'] = df['Magnitude']/2048

#     # Total Time and Sample Rate Calculation
#     df['Timestamp'] = pd.to_datetime(df['time'])
#     time = df['Timestamp'].values
#     sample_rate = int(np.round((time[1] - time[0]) / np.timedelta64(1, 'ms')))
#     total_time = int(np.round((time[-1] - time[0]) / np.timedelta64(1, 'ms')))
#     total_time_s = round(total_time/1000)

#     print("Sleep Calculations :")
#     print("Total Time : %ds ; %s hr " %(total_time_s, str(datetime.timedelta(seconds=total_time_s))))
#     print("Sample Rate: %ds" %(sample_rate))
#     sleep_data['total_time'] =  str(datetime.timedelta(seconds=total_time_s))
#     sleep_data['sample_rate'] =  sample_rate
    


#     # Bed Occupancy Calculations
#     if bedOccToggleOne == 1 and bedOccToggleTwo == 1:
#         df = df[(df['OcV'].astype(float) < bedOccThresholdOne) & (df['OcV2'].astype(float) < bedOccThresholdTwo)].reset_index()
#     elif bedOccToggleOne == 1:
#         df = df[df['OcV'].astype(float) < bedOccThresholdOne].reset_index()
#     elif bedOccToggleTwo == 1:
#         df = df[df['OcV2'].astype(float) < bedOccThresholdTwo].reset_index()
#     else:
#         # Both toggles are inactive
#         pass

#     time = df['Timestamp'].values
    
#     try :
#         bedOccupantTime = int(np.round((time[-1] - time[0]) / np.timedelta64(1, 'ms')))
#     except :
#         print("No Occupant Activity")
#         bedOccupantTime = 0

#     bedOccupantTime_s = round(bedOccupantTime/1000)
#     print("Bed Occupant Time : %ss ; %s hr " %(bedOccupantTime_s, str(datetime.timedelta(seconds=bedOccupantTime_s)) ))
#     sleep_data['bedOccupantTime'] =  str(datetime.timedelta(seconds=bedOccupantTime_s))

#     # Sleep Time and Awake Time Calculations
#     total_time = bedOccupantTime
#     magnitude = df['Magnitude'].values
#     threshold = 1.1


#     sleep_time = len(np.where(magnitude < threshold)[0]) * sample_rate // 1000
#     awake_time = total_time // 1000 - sleep_time

#     # magnitude_diff = np.abs(np.diff(magnitude))
#     # magnitude_diff[magnitude_diff < np.mean(magnitude_diff)] = 0
#     # magnitude_diff[magnitude_diff > 0] = 1
#     # move_timestamps = np.where(magnitude_diff == 1)[0] * sample_rate // 1000
#     # move_duration = np.median(np.diff(move_timestamps))
#     #move_freq = len(move_timestamps) / awake_time

#     #*******Movement Timestamp Record **********#
#     print("Getting Movement Instances")
#     print("print magnitude")

#     # Define the bins
#     bins = [0, 0.5, 0.8, 1, 1.2, float('inf')]

#     # Bin the values and count the number of values in each bin
#     magnitude_counts = pd.cut(df['Magnitude'], bins=bins).value_counts()

#     # Print the resulting distribution
#     print(magnitude_counts)
#     # less_than_threshold = df[df['Magnitude'] <0]
#     # timestamps = less_than_threshold.loc[:, 'time']
#     # print(timestamps)

#     # print(df['Magnitude'])
#     # indices = np.where(magnitude < threshold)[0]
#     # timestamps = df.iloc[indices]['time']
#     # print(timestamps)


#     print("Sleep Time : %ds ; %s hr " %(sleep_time, str(datetime.timedelta(seconds=sleep_time)) ))
#     print("Awake Time : %ds ; %s hr " %(awake_time, str(datetime.timedelta(seconds=awake_time)) ))

#     sleep_data['sleep_time'] =  str(datetime.timedelta(seconds=sleep_time))
#     sleep_data['awake_time'] =  str(datetime.timedelta(seconds=awake_time))

#     if total_time_s == 0 or awake_time+sleep_time == 0 :
#         print("In the if statement of Zero Checks")
#         print("Bed Occupant Time is", awake_time+sleep_time)
#         Sleep_info = "Summary Not Available"
    
#     else :
#         Sleep_info = "Of the" + " " + str(sleep_data['total_time']) + " " "of monitoring :" + "\n" + "You slept for" + " " + str(int((bedOccupantTime_s/total_time_s)*100)) + "%" + " " + "of the time." + "\n" + "Your pillow was unoccupied for" + " " + str(int(100- int((bedOccupantTime_s/total_time_s)*100))) +"%"+ " of the time." + "\n" + "You slept restfully for" + " " + str( int((sleep_time/(awake_time+sleep_time))*100) ) + "%" + " of the entire sleep duration."
    
#     sleep_data['Sleep_info'] = Sleep_info

#     print(Sleep_info)
#     # Get from front-end otherwise default as following :
#     rationIndexes = [0.95, 0.9, 0.8, 0.7]

#     Sleep_ratio, Sleep_star, Sleep_quality = getSleepRating(sleep_time, awake_time,rationIndexes)
    
#     print("Sleep Quality Outputs :")
#     print("Sleep Ratio :", Sleep_ratio)
#     print("Sleep Star :", Sleep_star)
#     print("Sleep Quality :", Sleep_quality)
    
#     sleep_data['Sleep_ratio'] = Sleep_ratio
#     sleep_data['Sleep_star'] = Sleep_star
#     sleep_data['Sleep_quality'] = Sleep_quality

#     print("Analysis Complete ...................................")

#     jsonapidata = json.dumps(sleep_data)

#     return jsonapidata

def algov2(df) :

    #df = pd.read_csv('./processedData.csv')
    #df.to_csv('./procssedSleepData.csv')


    print("Running the algorithm ...............")


    occToggle = 0
    occThresh = 1.2

    sleep_data = {}

    sample_rate = 250

    df = df[::-1]

    totalTimeHR, totalPoints = totalTime(df)

    print("Adding Magnitude...")
    df = addMagnitude(df)


    print("Computing Bed Occupancy...")
    df, sample_rate = bedOccData(df, occToggle, occThresh)


    print("Hour KPIs Computation...")
    totalOccPoints, countNegative, countPositive, totalMovingPoints = magArrayProcessing(df['Magnitude'].values)

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
    

    print()

    if sleep_data['bedOccupantTime'] == "00:00:00" :
        print("In the if statement of Zero Checks")
        print("Bed Occupant Time is", sleep_data['bedOccupantTime'])
        Sleep_info = "Summary Not Available"
    
    else :
        Sleep_info = "Of the" + " " + str(sleep_data['total_time']) + " " "of monitoring :" + "\n" + "Your bed was occupied for" + " " + str(int((totalOccPoints/totalPoints)*100)) + "%" + " " + "of the time." + "\n" + "Your pillow was unoccupied for" + " " + str(int(100- int((totalOccPoints/totalPoints)*100))) +"%"+ " of the time." + "\n" + "You slept restfully for" + " " + str( int(((totalOccPoints-totalMovingPoints)/(totalMovingPoints+(totalOccPoints-totalMovingPoints)))*100) ) + "%" + " of the entire sleep duration."
    
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

    jsonapidata = json.dumps(sleep_data)

    return jsonapidata

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
    df['OcV'] = df['OcV'].astype(float)/100

    # Bed Occupancy Calculations
    sample_rate = 250
    
    if (occToggle == 1) :
        df = df[df['OcV'].astype(float) < occThresh].reset_index()
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

def magArrayProcessing (array) :
    sample_rate = 250
    
    # Total 
    totalPoints = len(np.where(array < 100)[0]) * sample_rate // 1000
    
    # Calculate the average of the array
    avg = np.mean(array)

    # Set a threshold as the average
    thresholdPositive = avg+0.03
    thresholdNegative = avg-0.03

    # Count the number of values greater than the threshold
    countPositive = np.sum(array > thresholdPositive)
    countNegative = np.sum(array < thresholdNegative)
    
    totalMovingPoints = countPositive + countNegative
    
    totalMovingPoints = totalMovingPoints*sample_rate // 1000
    
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
    magBins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.94, 0.96, 0.97,1, 1.1, 1.2, 1.3, 1.4, 1.5, float('inf')]
    magCounts = pd.cut(df['Magnitude'], bins=magBins).value_counts()
    
    print("Magnitude Distribution")
    print(magCounts.sort_index())
    
    occVBins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, float('inf')]
    occVCounts = pd.cut(df['OcV'], bins=occVBins).value_counts()
    
    print("occV Distribution")
    print(occVCounts.sort_index())    

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

def Devicestatus(request):

    if request.method == "POST":
        sleep_data = {}
        jsondata = json.loads(request.body)
        print(jsondata)
        latest_datetime = SleepLabOptv1.objects.filter(DevID=jsondata).order_by('-timestamp').first()
        previous_datetime= datetime.datetime.strptime(latest_datetime.timestamp, '%Y-%m-%d %H:%M:%S.%f') 
        print('previous_datetime', previous_datetime)
        present_datetime = datetime.datetime.now()
        # Calculate the difference between the two dates
        time_difference = present_datetime - previous_datetime

        # Check if the difference is greater than 1 minute
        if time_difference > datetime.timedelta(minutes=1):
            data = {
                'Device': 'Not Active',
            }
        else:
            data['Device'] = 'Active'
        data['lastActive'] = previous_datetime

        obj= SleepLabOptv1.objects.filter(DevID=jsondata).order_by('-auto_increment_id')[0]

        obj_data = obj.jsonData

        print(type(obj.jsonData))

        print(obj_data['S239'])

        AcX = obj_data['S239']['AcX']
        AcY = obj_data['S239']['AcY']
        AcZ = obj_data['S239']['AcZ']

        if(int(AcX) > int(AcZ) or int(AcY)>int(AcZ)) :
            orientation = 'Vertical'
            data['orientation'] = 'Vertical'
        else :
            orientation = 'Horizontal'
            data['orientation'] = 'Horizontal'
        
    return JsonResponse(data)



def sleephighligh(request):
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, DataTable, TableColumn
    from bokeh.layouts import column
    import pandas as pd

    data = pd.read_csv('withmag.csv')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Oc2'].astype(int)
    df = data
    #     # create sample data
    # data = {'magnitude': [2.5, 3.2, 4.1, 3.5, 2.9],
    #         'timestamp': ['2023-04-26 08:00:00', '2023-04-26 08:01:00', '2023-04-26 08:02:00', '2023-04-26 08:03:00', '2023-04-26 08:04:00'],
    #         'avg': [0, 1, 0, 1, 0]}

    # # create DataFrame
    # df = pd.DataFrame(data)

    # display DataFrame
    print(df)

    source = ColumnDataSource(df)

    p = figure(title="Magnitude vs Timestamp", x_axis_type='datetime', x_axis_label='Timestamp', y_axis_label='Magnitude')

    # Add the line chart
    p.line(x='Timestamp', y='Magnitude', source=source, line_width=2)

    # Add a circle glyph for the points where avg == 1
    p.circle(x='Timestamp', y='Magnitude', source=df[df['Oc2'] == 1], fill_color='red', size=8)


    # # create figure
    # p = figure(title="Magnitude vs Timestamp", x_axis_label='Timestamp', y_axis_label='Magnitude')

    # # plot magnitude vs timestamp
    # p.line(df['Timestamp'], df['Magnitude'], line_width=2)

    # # highlight parts where avg is 1
    # # p.patch(df[df['Oc2'] == 1]['Timestamp'], df[df['Oc2'] == 1]['Timestamp'], alpha=0.3, color='red')
    # p.circle(x='Timestamp', y='Magnitude', source=source[df['avg'] == 1], fill_color='red', size=8)


    script, div = components(p)

    return render(request, 'sleeplabsgraph.html', {'script': script, 'div': div})



