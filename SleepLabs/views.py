from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from pytz import timezone
from SleepLabs.models import SleepLab
# Create your views here.

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


def sleep_labs_graph(request):
    x_axis = []
    acxdata = []
    acydata = []
    aczdata = []
    gyzdata = []
    gyxdata = []
    gyydata = []
    full_data =[]
    queryset = SleepLab.objects.all().order_by('timestamp')
    for sleeplabsdata in queryset:
        timedate = (sleeplabsdata.timestamp)
        timedate_asia = timedate.astimezone(timezone('Asia/Kolkata'))
        timedate_split = (str(timedate_asia)).split('.')
        x_axis.append(timedate_split[0])
        acxdata.append(int(sleeplabsdata.AcX))
        acydata.append(int(sleeplabsdata.AcY))
        aczdata.append(int(sleeplabsdata.AcZ))
        gyxdata.append(int(sleeplabsdata.GyX))
        gyydata.append(int(sleeplabsdata.GyY))
        gyzdata.append(int(sleeplabsdata.GyZ))
        dict_data = {'x_axis': timedate_split[0], 'acxdata' : int(sleeplabsdata.AcX), 'acydata' : int(sleeplabsdata.AcY), 
                    'aczdata' : int(sleeplabsdata.AcZ), 'gyzdata' : int(sleeplabsdata.GyZ), 'gyxdata' : int(sleeplabsdata.GyX), 
                    'gyydata' : int(sleeplabsdata.GyY)}
        full_data.append(dict_data) 

    return render(request, 'sleeplabs_graph.html', {
        'x_axis': x_axis,
        'acxdata' : acxdata,
        'acydata' : acydata,
        'aczdata' : aczdata,
        'gyzdata' : gyzdata,
        'gyxdata' : gyxdata,
        'gyydata' : gyydata,
        'sleeplabs': full_data
    })