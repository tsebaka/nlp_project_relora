import inspect
import importlib
from functools import reduce

adapter_types = ["lora"]

class Accuracy():
    def __init__(self):
        self.avg = 0
        self.reset()
    @property
    def accuracy(self):
        return self.avg
    def reset(self):
        self.count = 0
    def __call__(self, value):
        self.count += 1
        self.avg = (self.count * self.avg + value) / self.count
        return self.avg

def get_compatible_module_type(adapter_type: adapter_types):
    if adapter_type == "lora":
        module = importlib.import_module('peft.tuners.lora.layer')
        class_names = [name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__] 
        class_names += ["Linear8bitLt"]
        return [name for name in class_names if all(sub not in name for sub in ["Lora", "lora"])] 

def get_weight_names(model):
    unique_params = set()
    for name, param in model.named_parameters():
        name = name.rsplit(".", 2)
        name = name[-1] if name[-1] not in ["weight", "bias"] else name[-2]
        unique_params.add(name)
    return unique_params

def create_module_lists(model, adapter_type):
    module_list = get_compatible_module_type(adapter_type)
    adapter_modules, manual_modules, unique_names = [], [], set()
    
    for name, param in model.named_parameters():
        module_name = name.rsplit(".", 1)
        if module_name[1] in ["weight", "bias"]: 
            name = module_name[0] 
            module_name = name.rsplit(".", 1)
        module_name = module_name[-1] 

        if module_name not in unique_names: 

            unique_names.add(module_name)
            class_str = str(type(reduce(getattr, name.split('.'), model)))
            class_str = class_str.rstrip("'>")
            class_str = class_str.rsplit(".", 1)[-1]

            target_list = adapter_modules if class_str in module_list else manual_modules
            target_list.append(module_name)
            
    return adapter_modules, manual_modules