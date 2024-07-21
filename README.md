# Deep-Learning-Course


Ok the Chapters and course22p2-master directories are all of the notebooks that the teacher walked through in the course. I played around with them and you can go poke around there but that is where a lot of the code is broken. All of the main stuff that I've done is in the Projects folder.


## Projects
If you are planning on running any of these I would recommend making a new python virtual environment first. I'll add a requirements.txt that you can use to get the environment set up. 

### Kaggle
- Titanic
    - titanic_competition_nn.ipynb : This wa a kaggle competition I participated in. It was to predict the survivors on the titanic based on age, gender, class, etc. For the course we did this challenge with a neural net from sctach but I wanted to implement it using pytorch so thats what I did here. I think my best submission was 79% accuracy.

    - titanic_competition_xgb.ipynb : Everything is mostly the same as the last notebook. I just wanted to try XGBoost to see if it could do better than a neural net but it did worse. Maybe it could do better with some feature engineering.

- Digit Recognizer
    - minst_nn.ipynb : This is another challenge to recognize numbers from handwritten images. I implemented another neural network in pytorch and played around with the layer amount and size to get better accuracy. My best submission was 97% accuracy.

### Stable Diffusion
- For this project I wanted to learn a bit of streamlit since it was mentioned by the teacher. I also wanted to practice getting models from huggingface and running them on my local. A lot of the course focused on stable diffusion which was super cool. 
I built a little streamlit app that lets you compare two different models or the same two models and tweak some of the parameters. Then it generates the images and shows all of the intermediary images that the model made as well.
To run the app you navigate to the directory in the terminal then enter the command ``streamlit run app.py`` and it will open the app in your browser. I use smaller models but it still will take a few minutes to generate the images depending on your computer and the parameters. I think the longest its taken on my computer is like 10 mins. Let me know if you have issues and I can check it out. You also will probably need to get your own huggingface access token to download the models to run on your local if you want to try to get it running. Honestly it might be annoying to get it all set up so let me know if you just want me to show it to you over facetime.


