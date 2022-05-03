import discord
import numpy
import sys

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

client = discord.Client()

def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

def generate_impostor_message(messages):
    messages_str = "".join(messages)

    processed_inputs = tokenize_words(messages_str)

    chars = sorted(list(set(processed_inputs)))
    char_to_num = dict((c, i) for i, c in enumerate(chars))

    input_len = len(processed_inputs)
    vocab_len = len(chars)

    seq_length = 100
    x_data = []
    y_data = []

    for i in range(0, input_len - seq_length, 1):
        in_seq = processed_inputs[i:i + seq_length]

        out_seq = processed_inputs[i + seq_length]

        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])

    n_patterns = len(x_data)

    X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
    X = X/float(vocab_len)
    y = np_utils.to_categorical(y_data)

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X, y, epochs=20, batch_size=256)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    num_to_char = dict((i, c) for i, c in enumerate(chars))

    start = numpy.random.randint(0, len(x_data) - 1)
    pattern = x_data[start]

    formatted_message = ""

    random_character_amount = numpy.random.randint(50, 100)
    for i in range(random_character_amount):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocab_len)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = num_to_char[index]

        formatted_message += result

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return formatted_message

@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$impersonate'):
        print("Beginning impersonation")
        user_id = message.content.split(' ')[1]

        server_list = []
        for channel in message.guild.channels:
            if str(channel.type) == 'text':
                server_list.append(channel)


        text_message_list = []
        for server in server_list:
            messages = await channel.history(limit=300).flatten()
            text_message_list.extend(messages)

        message_content_by_user = []
        for message in text_message_list:
            if str(message.author.id) in user_id:
                message_content_by_user.append(message.content)

        impostor_message = generate_impostor_message(message_content_by_user)
        await message.channel.send(impostor_message)

# Replace with bot secret key
client.run('SECRET KEY HERE')




