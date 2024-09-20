import openai
import json


class GPT:
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        self.model = model

    def generate_user_profile(self, user_id: str, user_history_prompt: str) -> dict:
        prompt = 'Given the following check-ins of user {user_id}, generate a 200-word user profile summary to be used as a system prompt to another LLM that simulates this person\'s behavior, preferences, routines, hobbies, schedule, etc. Predict this user\'s Big Five Inventory traits: (1) extroverted / introverted, (2) agreeable / antagonistic, (3) conscientious / unconscientious, (4) neurotic / emotionally stable, (5) open / closed to experience. Also predict their age: child (<13) / adolescent (13-17) / adult (18-64) / older adult (>64); their gender: male / female; their educational background: some schooling / high school / college & beyond; and their socioeconomic level: lower / middle / upper. Also include any patterns that is observed and POI IDs of important places that might be visited in the future in the user profile summary. This system prompt will be used to make future check-in predictions. Return your response in JSON format: {{"traits": [trait1, trait2, trait3, trait4, trait5], "attributes": [age, gender, edu, socioeco], "preferences": [preference1, preference2, preference3, ...], "routines": [routine1, routine2, routine3, ...], "user_profile": "200-word user profile summary"}}\n{user_history_prompt}'

        response = openai.ChatCompletion.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(user_id=user_id, user_history_prompt=user_history_prompt),
                }
            ],
        )

        result = response.choices[0].message.content
        return json.loads(result)

    def generate_poi_intention(self, system_prompt: str, inputs: str, targets: str, **kwargs) -> dict:
        prompt = '{system_prompt}\n\n{trajectory}\n\n{targets}\n\nWith a step-by-step reasoning, think and suggest why they might have intended on visiting that POI. Analyze the user\'s profiles, previous visits in trajectory, time of visit of the POI, and their routines. DO NOT mention or analyze the selected POI id at all. You may suggest POI categories that they might have wanted to visit at that hour and based on their profile and trajectory. Return your response in JSON format: {{"profile_analysis": ..., "trajectory_analysis": ..., "time_of_visit_analysis": ..., "routines_and_preferences_analysis": ..., "potential_categories_of_interest_analysis": ..., "verdict": ...}}. All values should be in string format.'

        response = openai.ChatCompletion.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(system_prompt=system_prompt, trajectory=inputs, targets=targets),
                }
            ],
        )

        result = response.choices[0].message.content
        return json.loads(result)
