from pprint import pprint
import pkg_resources
import json
import core.functions

class Core:

    def __init__(self):
        # load brain.json file that contains how to think
        brain_file = pkg_resources.resource_string(__package__, 'brain.json')
        self.brain_cfg = json.loads(brain_file)
        self.functions = core.functions

    def process(self, data, utils):
        if data['message']['type'] == 'text':
            pprint(data['nlu']['intent'])
            reasoning = self.brain_cfg.get(data['nlu']['intent']['value'], None)
            if not reasoning:
                # TODO if only entities, this is a response to a previous question (if previous state saved)
                # then pop the state and continue processing
                reasoning = self.brain_cfg['other']

            # TODO check requirements
            # if requirements are missing, save state and ask user to fill requirements

            # TODO do steps (adding informations and other data)
            for step_descr in reasoning['steps']:
                step_fn = getattr(self.functions, step_descr['name'])
                # TODO invoke step_fn with required arguments
                #step_fn()

            data['decision'] = reasoning['output']

        elif data['message']['type'] == 'location':
            # TODO set user location and pop the previous state
            pass
