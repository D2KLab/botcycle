from pprint import pprint
import pkg_resources
import json
import core.functions

class Core:

    def __init__(self):
        # load brain.json file that contains how to think
        brain_file = pkg_resources.resource_string(__package__, 'brain.json').decode('utf-8')
        self.brain_cfg = json.loads(brain_file)
        self.functions = core.functions
        self.chat_contexts = {}
        #self.states = {}

    def process(self, data, utils):
        pprint(data['nlu']['intent'])

        # TODO using state machine to detect if new user has not yet set the required profile informations: [city]
        """
        user_state = self.states.get(data['chat_id'], None)
        if not user_state:
            data['decision'] = 'ask_city'
            return
        """

        # TODO use a function to check if the intent is not explicit but is consequence of a previous missing requirement
        chat_context = self.chat_contexts.get(data['chat_id'], None)
        if chat_context and chat_context['resume'] == 'todo':
            # this is a sub-interaction because some requirements were missing
            if self.fills_intent(data, chat_context):
                # the user utterance matches the requirements filling
                # TODO save somewhere the entity to be injected next
                # set a flag in the chat context. After handling the current utterance, if success the previous interaction will be resumed
                chat_context['resume'] = 'ready'
                print('ready to resume chat_context')

            else:
                # the user is jumping away of the previous unfilled request
                # delete the previous request
                self.chat_contexts.pop(data['chat_id'], None)


        intent = data['nlu'].get('intent', None)
        reasoning = self.brain_cfg.get(intent.get('value', None), None) if intent else None
        if not reasoning and not chat_context:
            # unexpected intent / no intent and no chat_context
            reasoning = self.brain_cfg['other']
            utils['generate_response'](data)
            return

        if reasoning:
            # TODO check requirements
            requirements = self.check_requirements(reasoning['requirements'], data, utils)
            # TODO if some requirements are missing, save state and ask user to fill requirements

            if requirements['ok']:
                # can proceed with steps
                for step_descr in reasoning['steps']:
                    step_fn = getattr(self.functions, step_descr['name'])
                    # invoke step_fn with required arguments
                    step_fn(data, utils, requirements)

                data['decision'] = reasoning['output']

            else:
                # TODO save the current state and set the latent intent for next sentences
                self.chat_contexts[data['chat_id']] = {'data': data, 'resume': 'todo', 'consequence': data['missing']}
                pass

            # generate the response
            utils['generate_response'](data)

        # TODO collect episodic data

        print('last part')
        pprint(chat_context)

        # check if the chat_context needs to be resumed
        if chat_context and chat_context['resume'] == 'ready':
            print('going to resume chat context')
            consequence = chat_context['consequence']
            # inject the previously missing requirement
            # TODO get the entity that was saved to be injected here
            chat_context['data']['nlu']['entities'][consequence['inject']] = data['nlu']['entities'].get('location', data['nlu']['entities']['user_position'])
            # check requirements again
            # do steps
            # provide answer
            del self.chat_contexts[data['chat_id']]
            self.process(chat_context['data'], utils)



    def check_requirements(self, requirements, data, utils):
        result = {}
        all_reqs_ok = True
        for requirement in requirements:
            entities = data['nlu']['entities']
            resolution = None
            for candidate in requirement['candidate_entities']:
                resolution = entities.get(candidate, None)
                if resolution:
                    break

            # check if the requirement is satisfied
            if not resolution:
                # if the requirement is not satisfied, turn off ok flag
                all_reqs_ok = False
                data['missing'] = requirement['consequence']
                data['decision'] = requirement['consequence']['action']
                # can exit loop since one requirement is not ok
                break

            print('requirement ' + requirement['name'] + ' = ' + str(resolution))
            result[requirement['name']] = resolution

        result['ok'] = all_reqs_ok

        return result

    def fills_intent(self, data, chat_context):
        if not data['nlu'].get('intent', None):
            # TODO save useful entity
            return True

        # TODO if intent matches, return true    
        return False
