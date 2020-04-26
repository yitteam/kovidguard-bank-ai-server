# API detect phishing bank - Hackcovy 2020

### Installing

Model đuợc chúng tôi train trong 2 ngày (chính xác 80%) google drive:https://drive.google.com/file/d/1mttArXJB-VTG53GuTS8vs46UKjgl6TmE/view?usp=sharing

```
pip install -r requirements.txt
```

## Running the app
Start Flask app
```
python app.py
```
wait until this appear 
```
Running on http://127.0.0.1:5000/
```
Everything is ready

### Test

Go to cd folder and test
curl -X POST -F image=@<image.jpg> "http://localhost:5000/predict_json?url="

example
```
cd "test data"
curl -X POST -F image=@ag.jpg "http://localhost:5000/predict_json?url="

reponse go like this
{
	result: "safe|phishing"
	origin: "somebank.com"  
}
```
## Authors

* **Grey Matter** Young IT AI Lab

