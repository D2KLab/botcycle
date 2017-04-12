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
        self.previous_state = {}

    def process(self, data, utils):
        pprint(data['nlu']['intent'])
        # check if the user is answering to a previous question
        prev = self.previous_state.get(data['chat_id'], None)
        # TODO find a way to enable the user to discard question and do a new one
        if prev:
            data['core_next'] = prev
            data['core_next']['enabled'] = False
            pass

        reasoning = self.brain_cfg.get(data['nlu']['intent']['value'], None)
        if not reasoning:
            # TODO if only entities, this is a response to a previous question (if previous state saved)
            # then pop the state and continue processing
            reasoning = self.brain_cfg['other']

        # TODO check requirements
        requirements = self.check_requirements(reasoning['requirements'], data, utils)
        # TODO if some requirements are missing, save state and ask user to fill requirements

        if requirements['ok']:
            # TODO do steps (adding informations and other data)
            for step_descr in reasoning['steps']:
                step_fn = getattr(self.functions, step_descr['name'])
                # TODO invoke step_fn with required arguments
                step_fn(data, utils, requirements)

            data['decision'] = reasoning['output']

        # TODO collect episodic data


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
                data['decision'] = requirement['consequence']
                # can exit loop since one requirement is not ok
                break

            print('requirement ' + requirement['name'] + ' = ' + str(resolution))
            result[requirement['name']] = resolution

        result['ok'] = all_reqs_ok

        if not all_reqs_ok:
            # save in the previous state
            self.previous_state[data['chat_id']] = data
        else:
            if self.previous_state.pop(data['chat_id'], None):
                data['core_next']['enabled'] = True

        return result
