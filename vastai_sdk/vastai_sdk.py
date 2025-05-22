import importlib
import types
import argparse
from typing import Optional, Any
import io
import contextlib
import requests
import inspect
import re
import os
import sys
import logging
from pyparsing import Word, alphas, alphanums, oneOf, Optional, Group, ZeroOrMore, quotedString, delimitedList, Suppress

from .vastai_base import VastAIBase
from .vast import parser, APIKEY_FILE
from . import vast as _vast
from textwrap import dedent

logging.basicConfig(level=os.getenv('LOGLEVEL') or logging.INFO)
logger = logging.getLogger()


_regions = {
  'AF': ('DZ,AO,BJ,BW,BF,BI,CM,CV,CF,TD,KM,CD,CG,DJ,EG,GQ,ER,ET,GA,GM,GH,GN,'
         'GW,KE,LS,LR,LY,MW,MA,ML,MR,MU,MZ,NA,NE,NG,RW,SH,ST,SN,SC,SL,SO,ZA,'
         'SS,SD,SZ,TZ,TG,TN,UG,YE,ZM,ZW'),  # Africa
  'AS': ('AE,AM,AR,AU,AZ,BD,BH,BN,BT,MM,KH,KW,KP,IN,ID,IR,IQ,IL,JP,JO,KZ,LV,'
         'LI,MY,MV,MN,NP,KR,PK,PH,QA,SA,SG,LK,SY,TW,TJ,TH,TR,TM,VN,YE,HK,'
         'CN,OM'),  # Asia
  'EU': ('AL,AD,AT,BY,BE,BA,BG,HR,CY,CZ,DK,EE,'
         'FI,FR,GE,DE,GR,HU,IS,IT,KZ,LV,LI,LT,'
         'LU,MT,MD,MC,ME,NL,NO,PL,PT,RO,RU,RS,'
         'SK,SI,ES,SE,CH,UA,GB,VA,MK'),  # Europe
  'LC': ('AG,AR,BS,BB,BZ,BO,BR,CL,CO,CR,CU,DO,EC,SV,GY,HT,HN,JM,MX,NI,PA,PY,'
         'PE,PR,RD,SUR,TT,UR,VZ'),  # Latin America and the Caribbean
  'NA': 'CA,US',  # Northern America
  'OC': ('AU,FJ,GU,KI,MH,FM,NR,NZ,PG,PW,SL,TO,TV,VU'),  # Oceania
}

def reverse_mapping(regions):
    reversed_mapping = {}
    for region, countries in regions.items():
        for country in countries.split(','):
            reversed_mapping[country] = region
    return reversed_mapping

_regions_rev = reverse_mapping(_regions)

def queryParser(kwargs, instance):
  # georegion uses the region modifiers as top level
  # descriptors
  #
  # chunked reduces values communicated to more usable chunks
  state = {'georegion': False, 'chunked': False }

  if kwargs.get('query') is not None: 
    qstr = kwargs['query']

    key = Word(alphas + "_-")
    operator = oneOf("= in != > < >= <=")
    single_value = Word(alphanums + "_.-") | quotedString

    array_value = (
        Suppress("[") + delimitedList(quotedString) + Suppress("]")
    ).setParseAction(lambda t: f"[{','.join(t)}]")
    value = single_value | array_value 
    expr = Group(key + operator + value)
    query = ZeroOrMore(expr)
    parsed = query.parseString(qstr)

    toPass = []

    for key in state.keys():
      state[key] = any([key, '=', 'true'] == list(expr) for expr in parsed)

    for expr in parsed:
      if expr[0] in state.keys():
        continue

      elif expr[0] == 'geolocation' and state['georegion']:
        region = _regions.get(expr[2].strip('"'))
        expr = ['geolocation', 'in', f'[{region}]']

      toPass.append(' '.join(expr))

    kwargs['query'] = ' '.join(toPass)

  return (state, kwargs)

def queryFormatter(state, obj, instance):
  # This algo is explicitly designed for skypilot to add 
  # depth to our catalog offerings
  cutoff = {
    'cpu_ram': 64 * 1024,
    'cpu_cores': 32,
    'min_bid': 0
  }

  upper = lambda amount: amount & (0xffff << max(amount.bit_length() - 1,1))

  filtered = []
  for res in obj:
    res['datacenter'] = (res['hosting_type'] == 1)
    if state['georegion'] and res['geolocation'] is not None:
      country = res['geolocation'][-2:]
      res['geolocation'] += f', {_regions_rev[country]}'

    if state['chunked']:
      good = True

      try:
        for k,v in cutoff.items():
          if res[k] is not None and (res[k] < cutoff[k]):
            good = False
          else:
            res[k] = cutoff[k]
      except:
        good = False

      if not good:
        continue

      #res['cpu_ram'] = upper(res['cpu_ram'])
      #res['cpu_cores'] = max(res['cpu_cores'] & 0xffff8, 4)
      res['gpu_ram'] = res['gpu_ram'] & 0xffffffffff0
      res['disk_space'] = int(res['disk_space']) & 0xffffffffffc0

    filtered.append(res)

  return filtered

def lastOutput(state, obj, instance):
    return instance.last_output
  
_hooks = {
    'search__offers': [queryParser, queryFormatter],
    'logs': [None, lastOutput],
    'execute': [None, lastOutput]
}

class VastAI(VastAIBase):
    """VastAI SDK class that dynamically imports functions from vast.py and binds them as instance methods."""

    def __init__(
        self,
        api_key=None,
        server_url="https://console.vast.ai",
        retry=3,
        raw=True,
        explain=False,
        quiet=False,
        curl=False
    ):
        if not api_key:
            if os.path.exists(APIKEY_FILE):
                with open(APIKEY_FILE, "r") as reader:
                    api_key = reader.read().strip()
                    self._creds = "FILE"
            else:
                self._creds = "NONE"
        else:
            self._creds = "CODE"

        self._KEYPATH = APIKEY_FILE
        self.api_key = api_key
        self.api_key_access = api_key
        self.server_url = server_url
        self.retry = retry
        self.raw = raw
        self.explain = explain
        self.quiet = quiet
        self.curl = curl
        self.imported_methods = {}
        self.last_output = None
        self.import_cli_functions()

    @property
    def creds_source(self):
        return self._creds

    def generate_signature_from_argparse(self, parser):
        parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        isFirst = True
        docstring = ''
        
        for action in sorted(parser._actions,  key=lambda action: len(action.option_strings) > 0):
            if action.dest == 'help':  
                continue
            if "Alias" in action.help:
                continue
            
            # Determine parameter kind
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
            if action.option_strings:
                kind = inspect.Parameter.KEYWORD_ONLY
            
            # Determine default and annotation
            default = action.default if action.default != argparse.SUPPRESS else None
            annotation = action.type if action.type else Any

            # Create the parameter
            param = inspect.Parameter(
                action.dest,
                kind=kind,
                default=default,
                annotation=annotation
            )
            parameters.append(param)

            if isFirst:
                docstring = 'Args:\n'
                isFirst = False

            param_type = annotation.__name__ if hasattr(annotation, "__name__") else "Any"
            help_text = f"{action.help or 'No description'}"
            docstring += f"\t{action.dest} ({param_type}): {help_text}\n"
            if default is not None:
                docstring += f"\t\tDefault is {default}.\n"

        # Return a custom Signature object
        sig = inspect.Signature(parameters)
        return sig, docstring

    def import_cli_functions(self):
        """Dynamically import functions from vast.py and bind them as instance methods."""

        if hasattr(parser, "subparsers_") and parser.subparsers_:
            for name, subparser in parser.subparsers_.choices.items():
                if name == "help":
                    continue
                if hasattr(subparser, "default") and callable(subparser.default):
                    func = subparser.default
                elif hasattr(subparser, "_defaults") and "func" in subparser._defaults:
                    func = subparser._defaults["func"]
                else:
                    print(
                        f"Command {subparser.prog} does not have an associated function."
                    )
                    continue

                func_name = func.__name__.replace("__", "_")
                wrapped_func = self.create_wrapper(func, func_name)
                setattr(self, func_name, types.MethodType(wrapped_func, self))
                arg_details = {}
                if hasattr(subparser, "_actions"):
                    for action in subparser._actions:
                        if action.dest != "help" and hasattr(action, "option_strings"):
                            arg_details[action.dest] = {
                                "option_strings": action.option_strings,
                                "help": action.help,
                                "default": action.default,
                                "type": str(action.type) if action.type else None,
                                "required": action.default is None and action.required,
                                "choices": getattr(
                                    action, "choices", None
                                ),  # Capture choices
                            }

                #globals()[func_name] = arg_details
                self.imported_methods[func_name] = arg_details
        else:
            print("No subparsers have been configured.")

    def create_wrapper(self, func, method_name):
        """Create a wrapper to check required arguments, convert keyword arguments, and capture output."""

        def wrapper(self, **kwargs):
            arg_details = self.imported_methods.get(method_name, {})
            for arg, details in arg_details.items():
                if details["required"] and arg not in kwargs:
                    raise ValueError(f"Missing required argument: {arg}")
                if (
                    arg in kwargs
                    and details.get("choices") is not None
                    and kwargs[arg] not in details["choices"]
                ):
                    raise ValueError(
                        f"Invalid choice for {arg}: {kwargs[arg]}. Valid options are {details['choices']}"
                    )
                kwargs.setdefault(arg, details["default"])

            kwargs.setdefault("api_key", self.api_key)
            kwargs.setdefault("url", self.server_url)
            kwargs.setdefault("retry", self.retry)
            kwargs.setdefault("raw", self.raw)
            kwargs.setdefault("explain", self.explain)
            kwargs.setdefault("quiet", self.quiet)
            kwargs.setdefault("curl", self.curl)

            # if we specified hooks we get that now
            state = None
            if func.__name__ in _hooks and _hooks[func.__name__][0] is not None:
              state, kwargs = _hooks[func.__name__][0](kwargs, self)

            args = argparse.Namespace(**kwargs)
            _vast.ARGS = args

            if logger.isEnabledFor(logging.DEBUG):
                kwargs_repr = {key: repr(value) for key, value in kwargs.items()}
                logging.debug(f"Calling {func.__name__} with arguments: kwargs={kwargs_repr}")
            else:
                out_b = io.StringIO()
                out_o = sys.stdout
                sys.stdout = out_b

            res = ''
            try:
                res = func(args) 
            except:
                pass

            if not logger.isEnabledFor(logging.DEBUG):
                sys.stdout = out_o
                self.last_output = out_b.getvalue()
                out_b.close()

            if func.__name__ in _hooks:
              res = _hooks[func.__name__][1](state, res, self)

            if hasattr(res, 'json'):
               logging.debug(f" └-> {res.json()}")
               return res.json()

            logging.debug(f" └-> {res}")

            return res

        func_name = func.__name__.replace("__", "_")
        wrapper.__name__ = func_name

        wrapper.__doc__ = ''
        hasDoc = False
        # We don't want to be redundant so we look for help in various places and 
        # if it's not empty after we parse through it then we use it as our
        # canonical help. So we go in this order:
        #
        #   func.__doc__
        #   sig.epilog
        #   sig.help
        #

        if func.__doc__:
            doc = dedent(re.sub(r'\s(:param|@).*', '', func.__doc__, flags=re.DOTALL)).strip()
            if doc:
               hasDoc = True
               wrapper.__doc__ += f"{doc}\n\n"

        sig = getattr(func, "mysignature", None)
        sig_help = getattr(func, "mysignature_help", None)
        if sig:
            wrapper.__signature__, docappend = self.generate_signature_from_argparse(sig)
            epi = None

            if sig.epilog:
                epi = re.sub('Example.?:.*', '', sig.epilog, flags=re.DOTALL|re.M).strip()
                wrapper.__doc__ += epi

            if not (epi or hasDoc) and sig_help:
                wrapper.__doc__ += sig_help
            
            wrapper.__doc__ = '\n\n'.join([ wrapper.__doc__.rstrip(), docappend ])
        return wrapper

    def credentials_on_disk(self):
        """
        nop is the classic "no operation". This is just used to make sure the
        libraries don't crash and a key file exists
        """
        pass

    def __getattr__(self, name):
        if name in self.imported_methods:
            return getattr(self, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")

