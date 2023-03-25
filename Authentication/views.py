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
            return redirect('/home') 
        else:    
            messages.error(request, 'Username OR password is incorrect!')
    return render(request, 'login.html')


def logout(request):
	UserLogout(request)
	return redirect('login')  