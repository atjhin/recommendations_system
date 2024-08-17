# Song recommendations system

## Description
The goal of this project is to create an app where you can input a particular song and determine which of my playlist would this song belong to. As an avid music enjoyer, I am interested in exploring how statistical models and concepts can understand our music taste by analyzing the music's different features.

### Features
For more information on the following features check [here](https://developer.spotify.com/documentation/web-api/reference/get-audio-features).
- danceability
- energy 
- key 
- loudness 
- mode 
- speechiness
- acousticness
- instrumentalness 
- liveness 
- valence 
- tempo        
- duration_ms 
- time_signature

### Target
- Genre: Here genre is obtained from my playlists. Since my playlists are more or less grouped based on the genre of the song, this works well in categorizing them into genres. However, this is different from just predicting the genre of the song since a song genre is quite diverse (e.g. there are many type of pop songs, some are slower or more accousitic, some are more upbeat), I want to find which songs matches my definition of genre based on how I created my different playlists. The genres included will be:
    - Rap (rap songs similar to 21 and JID's style)
    - RnB (alternative r&b, Doja Cat and Alina Baraz)
    - Classical (mostly solo piano/violin and concerto pieces, Chopin and Tchaikovsky) 
    - Covers (accoustic covers of pop songs, Kina Grannis)
    - EDM (old EDM songs, Skrillex and Tiesto)
    - Old (old pop/jazz)
    - Easy (pop slow, Bruno Major and Luke Chiang)

## Procedure

- **Data Pre-process**: Extract data from spotify and create our training data (songs in our playlist), and testing data (other songs not in playlist). We will be creating our data sets by extracting features of songs from the spotify API, and defining the target variable (genre), based on my playlists.
    - **Training data**: We will chosing some of my playlists from spotify, obtain the audio features and group the playlists into particular genres labeled above.
    - **Testing data**: We will select random songs from created playlists that are not in the training data but similar to songs in the playlist and label them as those in the playlist. Since we are randomly picking songs, they might be misslabeled (i.e. the songs labeled as a particular genre might not be similar to the songs in it).
    
- **Modeling**: We will be using the sklearn module and consider the following models: Logistic regression, KNN, XGBoost and SVM. In each model, we will be considering the following methodologies depending on suitability in each case:
    - Scaling feature and target variables
    - Hyper parameter tuning
    - Feature selection
    - Model selection
    - Evaluating errors in testing data
    
    Other things to consider:
    - Overfitting: We want our models to understand the behaviour of our training data set as much as possible as our goal is to match the preferences of the individual. Hence, overfitting might be beneficial in this case. However, we want the model to still generate some possibilities of recommending different songs that might be suitable in that playlists. Therefore, we need to have a good balance of variance and bias while giving more weight in lower bias and higher variance.

- **Web App**: After choosing our final model, we will be creating a web application using flask to allow us to easily input songs and determine which playlist would these songs belong in.

## Project Structure

- TBD

## Usage Instructions

- TBD

## Future Works

- TBD

## Contributors

- **Alexander Michael Tjhin** - *Project Owner & Contributor*
    - Class of 2025 University of Waterloo student, responsible for project management and key development tasks.
    - (Email)[atjhin@uwaterloo.ca]
    - (Linkedin)[https://www.linkedin.com/in/atjhin/]
    - (Github) [https://github.com/atjhin]
    
- **Alexander Vincent Tjhin** - *Contributor*
    - Class of 2025 University of Toronto student, equally contributing to development and implementation of project.
    - (Email)[vincent.tjhin@mail.utoronto.ca]
    - (Linkedin)[https://www.linkedin.com/in/alexander-vincent-tjhin/]
    - (Github) [https://github.com/avincenttjhin]

## Resources

- **Spotify API Documentation**: [Spotify for Developers](https://developer.spotify.com/documentation/web-api/)
- **Scikit-Learn Documentation**: [Scikit-Learn Documentation](https://scikit-learn.org/stable/)