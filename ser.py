import streamlit as st 
import soundfile as sf
import io
import keras
import numpy as np
import librosa
import matplotlib.pyplot as plt



st.title('Speech Emotion Recognition')

uploaded_file=st.file_uploader('Choose Audio file',type=['wav','ogg'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    print(uploaded_file.name)
    st.audio(bytes_data, format='audio/ogg')
    scipy.io.wavfile.write(uploaded_file, 16000, bytes_data)
    data, sampling_rate = sf.read(io.BytesIO(bytes_data))
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(x, axis=0)
    loaded_model = keras.models.load_model('SER_model.h5')
    predict_x=loaded_model.predict(x)
    classes_x=np.argmax(predict_x,axis=1)
    label_conversion = {'0': 'Neutral',
                            '1': 'Calm',
                            '2': 'Happy',
                            '3': 'Sad',
                            '4': 'Angry',
                            '5': 'Fearful',
                            '6': 'Disgust',
                            '7': 'Surprised'}

    for key, value in label_conversion.items():
        if int(key) == classes_x:
            label = value 
    st.header('Predicted Emotion:',)
    st.subheader(label)

    fig, ax = plt.subplots()
     ax.bar(list(label_conversion.values()), predict_x[0])
     ax.set_xlabel('Emotion')
     ax.set_ylabel('Probability')
     ax.set_title('Emotion Recognition Results')
     st.pyplot(fig)

    # Chatbot for sad emotion
    if label == 'Sad':
        st.subheader('Chatbot')
        st.write('It seems like you are feeling sad. Here are some uplifting messages and activities that might help:')
        messages = ['You are not alone. Reach out to a friend or family member for support.',
                    'Take some time to practice self-care, such as taking a relaxing bath or going for a walk outside.',
                    'Listen to some uplifting music or watch a funny movie to help improve your mood.',
                    'Remember that it is okay to not be okay. Take it one day at a time and be kind to yourself.',
                    'Try doing something creative, like painting or writing, to help express your emotions.',
                    'Practice gratitude by writing down three things you are thankful for each day.',
                    'Remember that emotions are temporary and that things will get better.']
        message = np.random.choice(messages)
        st.write(message)

    if label == 'Happy':
        st.subheader('Chatbot')
        st.write('It seems like you are feeling sad. Here are some uplifting messages and activities that might help:')
        messages = ['Im so happy to hear that you are feeling happy! Is there anything specific thats contributing to your positive mood?',
                    'Thats great news! Remember to celebrate the little victories and spread positivity to those around you.'],
        message = np.random.choice(messages)
        st.write(message)

    if label == 'Calm':
        st.subheader('Chatbot')
        st.write('It seems like you are feeling sad. Here are some uplifting messages and activities that might help:')
        messages = ['I amm glad to hear that you are feeling calm. Let me know if there anything else you need.',
             'Remember to take breaks and practice self-care to help manage stress and anxiety.']
        message = np.random.choice(messages)
        st.write(message)

    if label == 'Fearful':
        st.subheader('Chatbot')
        st.write('It seems like you are feeling sad. Here are some uplifting messages and activities that might help:')
        messages = ['I am sorry to hear that you are feeling fearful. Is there anything specific thats causing your fear?',
                'Remember to take deep breaths and focus on the present moment to help manage anxiety.',
                'Let me know if theres anything I can do to help you feel more comfortable.']
        message = np.random.choice(messages)
        st.write(message)

    if label == 'Angry':
        st.subheader('Chatbot')
        st.write('It seems like you are feeling sad. Here are some uplifting messages and activities that might help:')
        messages = ['It sounds like you are feeling angry. Would you like some tips on how to manage your anger?',
              'Remember to take a break and cool down when you are feeling overwhelmed or frustrated.',
              'I am here to listen if you need to vent or talk through your feelings.']
        message = np.random.choice(messages)
        st.write(message)

    if label == 'Disgust':
        st.subheader('Chatbot')
        st.write('It seems like you are feeling sad. Here are some uplifting messages and activities that might help:')
        messages = ['It sounds like you are feeling disgusted. Is there anything specific thats causing your negative emotions?',
                'Remember to practice self-care and take breaks when you are feeling overwhelmed or stressed.',
                'I am here to listen if you need to talk about your feelings.']
        message = np.random.choice(messages)
        st.write(message)

    if label == 'Suprised':
        st.subheader('Chatbot')
        st.write('It seems like you are feeling sad. Here are some uplifting messages and activities that might help:')
        messages = ['Wow, thats surprising! Is there anything specific thats causing your surprise?',
                  'Remember to take some time to process the surprise and reflect on how you can respond.',
                  'Let me know if theres anything I can do to help you process your emotions.']
        message = np.random.choice(messages)
        st.write(message)

