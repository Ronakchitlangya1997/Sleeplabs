from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from pytz import timezone
from SleepLabs.models import SleepLab
import json
import datetime
import numpy as np
import math
# Create your views here.

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
def sleep_labs_graph(request):
    # full_data =[]
    # # queryset = SleepLab.objects.filter(timestamp__date = datetime.date(2021, 4, 7)).order_by('timestamp')
    # df = pd.DataFrame(list(SleepLab.objects.filter(timestamp__date = datetime.date(2023, 2, 11)).order_by('timestamp').values()))
    # #print(df)
    # # df = df1.head()
    # # df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
    # # print(df)
    # x = []
    # df['AcX'] = df['AcX'].astype('int')
    # df['AcY'] = df['AcY'].astype('int')
    # df['AcZ'] = df['AcZ'].astype('int')
    # for index, row in df.iterrows():
    #     x.append(math.sqrt(row['AcX']**2 + row['AcY']**2 + row['AcY']**2))
    # df['Mag'] = x
    # df['accel_mag'] = (df['Mag'] / 2048).round(2)
    # # df = df[df['accel_mag'] > 0.5]
    # for index, row in df.iterrows():
    #     timedate = str(row['timestamp'])
    #     split_timedate = timedate.split('.')
    #     dict_data = {'x_axis': split_timedate[0], 'acxdata' : int(row['AcX']), 'acydata' : int(row['AcY']), 
    #                 'aczdata' : int(row['AcZ']), 'gyzdata' : int(row['GyZ']), 'gyxdata' : int(row['GyX']), 
    #                 'gyydata' : int(row['GyY']), 'occdata': int(row['OCC']), 'accmag': row['accel_mag']}
    #     full_data.append(dict_data) 

    return render(request, 'sleeplabs_graph.html')


def sleep_labs_graph_api(request):
    if request.method == "POST":
        sleep_data = {}
        jsondata = json.loads(request.body)
        if(jsondata['Date']) == 'today':
            Date = datetime.date.today()
        else:
            Date = jsondata['Date']
        full_data =[]
        print(Date)

        df = pd.DataFrame(list(SleepLab.objects.filter(timestamp__date = Date).order_by('timestamp').values()))
        print(df)
        x = []
        df['AcX'] = df['AcX'].astype('int')
        df['AcY'] = df['AcY'].astype('int')
        df['AcZ'] = df['AcZ'].astype('int')

        for index, row in df.iterrows():
            x.append(math.sqrt(row['AcX']**2 + row['AcY']**2 + row['AcY']**2))
            
        df['Magnitude'] = x
        df['Magnitude'] = df['Magnitude']/2048

        # Convert the datetime column to a datetime object
        df['Timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract the time data from the DataFrame
        time = df['Timestamp'].values

        # Calculate the sample rate and total time
        sample_rate = int(np.round((time[1] - time[0]) / np.timedelta64(1, 'ms')))
        total_time = int(np.round((time[-1] - time[0]) / np.timedelta64(1, 'ms')))
        print(" total_time : %sms ; %shr: " %(total_time, round ((total_time/(60*60*1000)),2)))
        sleep_data['total_time'] = round((total_time/(60*60*1000)),2)
        
        df = df[df['OCC'].astype(int) == 1].reset_index()
        # Extract the time data from the DataFrame
        time = df['Timestamp'].values

        # Calculate the sample rate and total time
        sample_rate = int(np.round((time[1] - time[0]) / np.timedelta64(1, 'ms')))
        total_time = int(np.round((time[-1] - time[0]) / np.timedelta64(1, 'ms')))

        #Calculate the sleep time and awake time
        magnitude = df['Magnitude'].values
        threshold = np.mean(magnitude) + 0.5 * np.std(magnitude)

        sleep_time = len(np.where(magnitude < threshold)[0]) * sample_rate // 1000
        awake_time = total_time // 1000 - sleep_time

        # Calculate the movement duration, frequency, and timestamps
        magnitude_diff = np.abs(np.diff(magnitude))
        magnitude_diff[magnitude_diff < np.mean(magnitude_diff)] = 0
        magnitude_diff[magnitude_diff > 0] = 1
        move_timestamps = np.where(magnitude_diff == 1)[0] * sample_rate // 1000
        move_duration = np.median(np.diff(move_timestamps))
        move_freq = len(move_timestamps) / awake_time

        # # Print the results
        print("Bed_Occupancy_total_time : %sms ; %shr: " %(total_time, round ((total_time/(60*60*1000)),2)))
        print(" sample_rate :", sample_rate)


        print(" sleep_time : %ds ; %shr: " %(sleep_time, str(datetime.timedelta(seconds=sleep_time)) ))
        print(" awake_time : %ds ; %shr: " %(awake_time, str(datetime.timedelta(seconds=awake_time)) ))
        print(" move_duration : %ds ; %dhr: " %(move_duration, move_duration/(60*60) ))
        print(" move_freq:", move_freq)
        sleep_data['Bed_Occupancy_total_time'] = round ((total_time/(60*60*1000)),2)
        sleep_data['sample_rate'] = sample_rate
        sleep_data['sleep_time'] = str(datetime.timedelta(seconds=sleep_time))
        sleep_data['awake_time'] = str(datetime.timedelta(seconds=awake_time))
        sleep_data['Sleep Score'] = sleep_time/awake_time
        
        jsonapidata = json.dumps(sleep_data)
        return HttpResponse(jsonapidata)
    return HttpResponse('not working')

  #print(" move_timestamps (s): ", move_timestamps)

        # move_timestamps_hrs = move_timestamps / 3600
        # my_formatted_list = [ '%.2f' % elem for elem in move_timestamps_hrs ]
        # print(" move_timestamps (hrs) :", my_formatted_list)
        # for index, row in df.iterrows():
        #     x.append(math.sqrt(row['AcX']**2 + row['AcY']**2 + row['AcY']**2))
        # df['Mag'] = x
        # df['accel_mag'] = (df['Mag'] / 2048).round(2)
        # df = df[df['accel_mag'] > 1]
        # print(df)
        # for index, row in df.iterrows():
        #     timedate = str(row['timestamp'])
        #     split_timedate = timedate.split('.')
        #     dict_data = {'x_axis': split_timedate[0], 'acxdata' : int(row['AcX']), 'acydata' : int(row['AcY']), 
        #                 'aczdata' : int(row['AcZ']), 'gyzdata' : int(row['GyZ']), 'gyxdata' : int(row['GyX']), 
        #                 'gyydata' : int(row['GyY']), 'occdata': int(row['OCC']), 'accmag': row['accel_mag']}
        #     full_data.append(dict_data)