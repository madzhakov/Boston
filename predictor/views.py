from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
import numpy as np
import pickle

class PredictView(APIView):
    def post(self, request):
        data = request.data
        model_path = 'predictor/model.pkl'
        model = pickle.load(open(model_path, 'rb'))
        prediction = model.predict(np.array([list(data.values())]))
        return Response({"prediction": prediction.tolist()})