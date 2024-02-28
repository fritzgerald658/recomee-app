from django.shortcuts import render
from django.http import HttpResponse
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def get_started(request):
    if request.method == 'POST':
        # Get the username from the form
        username = request.POST.get('username')
        
        # Pass the username to the template
        return render(request, 'predict_position.html', {'name': username.upper()})

    return render(request, 'get_started.html')
    
def predict_position(request):
    if request.method == 'POST':
        # Get user input from the form
        user_course = request.POST.get('course')
        user_skills = request.POST.get('skills')
        user_interest = request.POST.get('interest')
        user_industry = request.POST.get('industry')
        # Combine user input into a single string
        user_input_combined = ' '.join([user_course, user_skills, user_interest, user_industry])

        # Load the model and vectorizer
        vectorizer = load('thesisapp/models/count_vectorizer.joblib')
        nb_model = load('thesisapp/models/career_recommender_ml_model.joblib')

        # Transform user input into numerical format
        user_input_vec = vectorizer.transform([user_input_combined])

        # Make a prediction and get probability estimates
        prediction_probabilities = nb_model.predict_proba(user_input_vec)[0]

        # Get the indices of the top three predictions
        top_three_indices = prediction_probabilities.argsort()[-3:][::-1]

        # Get the top three predictions and their probabilities
        top_three_predictions = nb_model.classes_[top_three_indices]
        top_three_probabilities = prediction_probabilities[top_three_indices]

        #Get top one result
        recomee_result_one = top_three_predictions[0]
        recomee_probability_one = top_three_probabilities[0]
        recomee_percentage_one = '{:.2f}'.format(recomee_probability_one * 100)
        
        #Get top two result
        recomee_result_two = top_three_predictions[1]
        recomee_probability_two = top_three_probabilities[1]
        recomee_percentage_two = '{:.2f}'.format(recomee_probability_two)

        # Get top three result 
        recomee_result_three = top_three_predictions[2]
        recomee_probability_three = top_three_probabilities[2]
        recomee_percentage_three = '{:.2f}'.format(recomee_probability_three)
        
        # Assigning results in a variable to be use for html
        top_one = {'first_career' : recomee_result_one.upper(), 'first_probability': recomee_percentage_one}
        top_two = {'second_career' : recomee_result_two.upper(), 'second_probability' : recomee_percentage_two}
        top_three = {'third_career' : recomee_result_three.upper(), 'third_probability' : recomee_percentage_three}

        # Combining results into one dictionary    
        combined_results = {**top_one, **top_two, **top_three}
        
        # Pass the top three results to the template
        return render(request, 'prediction_result.html', combined_results)
        

    return render(request, 'predict_position.html')
