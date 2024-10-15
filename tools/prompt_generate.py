import json
import os
import pickle
from openai import OpenAI
import base64

json_path = 'test_meta.json'
number = 1 #问题数量
client = OpenAI(api_key="yourkey")
processed_folders = []
processed_failed_folders = []
data_folder = 'test' #子文件夹名称
messages_setting = [
    {"role": "system", "content": """You are an extremely perceptive action analysis expert who is proficient in conducting in-depth and comprehensive analyses of people's actions. Based on the given images and text, you can propose novel questions of specific types and generate answers in the form as in the given example, showing a step-by-step reasoning process in the answer. In the questions and answers you provide, the actions should be related solely to body movements."""},
]

with open('user_input_list.pkl', 'rb') as f:
    user_input_list = pickle.load(f)


def image_content(file_folder):
    photo_list = []
    for filename in os.listdir(os.path.join( file_folder, 'masks')):
        if filename.endswith('.png'):
            photo_list.append(filename)
    print(os.path.join( file_folder, 'masks'))
    for i in range(4):
        num = i * int(len(photo_list)/4-0.5)
        image_path = os.path.join(file_folder,photo_list[num])
        base64_image = encode_image(image_path.replace('png', 'jpg'))
        content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
        })

            

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def cal_photonum(folder_path):
    jpg_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            jpg_count += 1
    return jpg_count

if os.path.exists('processed_folders.pkl'):
    """检查记录文件是否存在"""
    with open('processed_folders.pkl', 'rb') as f:
        processed_folders = pickle.load(f)

if os.path.exists('processed_failed_folders.pkl'):
    """检查记录错误文件是否存在"""
    with open('processed_failed_folders.pkl', 'rb') as f:
        processed_failed_folders = pickle.load(f)




with open(json_path, 'r') as f:
    data = json.load(f)
n = 0
for file_foleder in data['videos']:
    questions = data['videos'][file_foleder]['questions']
    questions = str(questions).replace("'", '"')

    #print(cal_photonum(os.path.join('test', file_foleder)))
    if file_foleder not in processed_folders:
        message = [{"role": "system", "content": """You are an extremely perceptive action analysis expert who is proficient in conducting in-depth and comprehensive analyses of people's actions. Based on the given images and text, you can propose novel questions of specific types and generate answers in the form as in the given example, showing a step-by-step reasoning process in the answer. In the questions and answers you provide, the actions should be related solely to body movements."""}]
      
        for i in range(len(user_input_list)):
            # Get user input
            #user_input = input("You: ")
            user_input = user_input_list[i]
            # Check if the user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation.")
                break
            content = [{"type" : "text", "text": user_input}]

            image_content(os.path.join(data_folder, file_foleder))

            content.append({
                "type": "text",
                "text": questions
                })
            # Add user input to the messages list
            
            message.append({"role": "user", "content": content})
            #print(messages)

            # Get the response from the model
            print("connecting to gpt-4o")
            messages = message
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    stream=False,
                    messages=messages
                )
                message.pop()
            except:
                print("connection failed")
                processed_failed_folders.append(file_foleder)
                with open('processed_failed_folders.pkl','wb') as f:
                    pickle.dump(processed_failed_folders, f)
                break
            # Extract and display the assistant's reply
            reply = completion.choices[0].message.content
            #print(f"Assistant: {reply}")
            print(completion.choices[0].message.content)

            # Add the assistant's reply to the conversation history
            message.append({"role": "assistant", "content": reply})
            

            print("-----------")
            print("len(messages):",len(message))

            if os.path.exists(os.path.join('output',f'{file_foleder}.txt')):
                with open(os.path.join('output',f'{file_foleder}.txt'), 'a',encoding='utf-8') as f:
                    f.write(completion.choices[0].message.content + '\n'+'---------\n')

            else:
                with open(os.path.join('output',f'{file_foleder}.txt'), 'w',encoding='utf-8') as f:
                    f.write(completion.choices[0].message.content + '\n'+'---------\n')



        processed_folders.append(file_foleder)

        with open('processed_folders.pkl','wb') as f:
            pickle.dump(processed_folders, f)

        n += 1
        print(f'{n} folders have been processed')
        if n == number:
            break
       

