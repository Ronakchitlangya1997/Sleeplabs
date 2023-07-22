from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from .models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login as UserLogin, logout as UserLogout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

# Email id function 
def get_user(email):
    try:
        return User.objects.get(email=email.lower())
    except User.DoesNotExist:
        return None

@csrf_exempt
def login(request):
    # user = User.objects.create_user(username='Johddfnson',
    #                              email='johncvdsonhermann@gmail.com',
    #                              password='Johnson', is_agent = True, is_admin = False)
    if request.method == 'POST':
        email = request.POST.get('emailid')
        password =request.POST.get('password')
        # print(password, email)
        username = get_user(email)
        # print(username)
        user = authenticate(request, username=username, password=password)
        # print(user)
        if user is not None:   
            UserLogin(request, user)
            return redirect('/') 
        else:    
            messages.error(request, 'Username OR password is incorrect!')
    return render(request, 'login.html')


def logout(request):
	UserLogout(request)
	return redirect('login')  

import json
 
from django.http import JsonResponse



import json
from django.core.serializers.json import DjangoJSONEncoder

# Assuming 'User' is the model representing your user data
from .models import User

class CustomJSONEncoder(DjangoJSONEncoder):
    def default(self, obj):
        if isinstance(obj, User):
            return obj.__dict__
        return super().default(obj)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def getUserMobileApp(request):
    if request.method == "POST":
        jsondata = json.loads(request.body)
        print("Printing JSON:")
        print(jsondata)
        device_id = jsondata.get('DeviceID', None)

        if device_id is not None:
            try:
                user = User.objects.get(Device_name=device_id)
                # Assuming 'User' has fields 'username' and 'email'.
                # You can modify this based on your actual model fields.
                user_data = {
                    "username": user.username,
                    "email": user.email
                }
                return JsonResponse(user_data)
            except User.DoesNotExist:
                return JsonResponse({"message": "User not found."}, status=404)
        else:
            return JsonResponse({"message": "Invalid JSON data or 'DeviceID' not provided."}, status=400)

    return JsonResponse({"message": "Invalid request method."}, status=405)
