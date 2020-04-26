import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict_json?url=https://www.google.com/"



# submit the request
r = requests.post(KERAS_REST_API_URL).json()
# print(r)
# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    # for (i, result) in enumerate(r):
    #     print(i,result)
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["website"],
            result["password-form"]))

# otherwise, the request failed
else:
    print("Request failed")