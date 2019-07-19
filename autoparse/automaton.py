import itertools

numbers = set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])


def preprocess(doc: str):
    if not isinstance(doc, str):
        return "<not-a-string>"
    preprocessed_doc = ""
    number_counter = 0
    doc = doc.lower()
    for char in doc:
        if not char in numbers:
            if number_counter == 0:
                preprocessed_doc += char
            else:
                preprocessed_doc += "<" + str(number_counter) + "-number>" + char
                number_counter = 0
        else:
            number_counter += 1
    if number_counter > 0:
        preprocessed_doc += "<" + str(number_counter) + "-number>"
    return preprocessed_doc


def match_generic(word: str, **kwargs):
    return word, None, kwargs["w"] != "<stop>"


def match_with_number(word: str, **kwargs):
    preprocessed_word = preprocess(word)
    return None, None, preprocessed_word == kwargs["w"]


def match_with_variable(word: str, **kwargs):
    return word, kwargs["var"], kwargs["w"] != "<stop>"


def match_default(word: str, **kwargs):
    return None, None, word == kwargs["w"]


class SimpleState:
    def __init__(self, state_id: int):
        self.id = state_id
        self.next = set()

    def add_transition(self, transition):
        self.next.add(transition)


class SimpleTransition:
    def __init__(self, state_out, word, tid, probabilities={}):
        self.word = word
        self.state_out = state_out
        self.tid = tid
        self.p = probabilities
        if word == "*":
            self.match = match_generic
            self.kwargs = {"w": word}
            self.type = "generic"
        elif "number>" in word:
            self.match = match_with_number
            self.kwargs = {"w": word}
            self.type = "number"
        elif "<$" in word:
            self.match = match_with_variable
            self.kwargs = {"w": word, "var": word.replace("<$", "").replace(">", "")}
            self.type = "var"
        else:
            self.match = match_default
            self.kwargs = {"w": word}
            self.type = "word"

    def matching_probability(self, history):
        if not history in self.p:
            return 0
        return self.p[history]

    def __hash__(self):
        return self.tid

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.tid == other.tid


class ActiveState:
    def __init__(
        self, state, captured_words=(), variables={}, probability: float = 1, history=()
    ):
        self.captured_words = captured_words
        self.variables = variables
        self.state = state
        self.probability = probability
        self.history = history

    def next_states(self, word):
        new_states = set()
        for transition in self.state.next:
            captured_word, var, success = transition.match(word, **transition.kwargs)
            if success:
                new_probability = self.probability * transition.matching_probability(
                    self.history
                )
                if transition.state_out.id == self.state.id:
                    new_history = tuple(
                        itertools.chain(self.history[1:], [self.history[-1]])
                    )
                else:
                    new_history = tuple(
                        itertools.chain(self.history[1:], [transition.tid])
                    )
                new_active_state = None
                new_vars = self.variables.copy()
                if var != None:
                    new_vars[var] = captured_word
                if not captured_word == None:
                    new_active_state = ActiveState(
                        transition.state_out,
                        tuple(itertools.chain(self.captured_words, [captured_word])),
                        variables=new_vars,
                        probability=new_probability,
                        history=new_history,
                    )
                else:
                    new_active_state = ActiveState(
                        transition.state_out,
                        self.captured_words,
                        variables=new_vars,
                        probability=new_probability,
                        history=new_history,
                    )
                new_states.add(new_active_state)
        return new_states

    def __hash__(self):
        return hash(self.state.id) ^ hash(self.history) ^ hash(self.captured_words)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.variables == other.variables
            and self.state.id == other.state.id
            and self.history == other.history
            and self.captured_words == other.captured_words
        )


class Automaton:
    def __init__(self, order: int = 0):
        self.start_state = SimpleState(0)
        self.stop_state = SimpleState(1)
        self.states = {0: self.start_state, 1: self.stop_state}
        self.order = order

    def add_state(self, state_id: id):
        if not state_id in self.states:
            new_state = SimpleState(state_id)
            self.states[state_id] = new_state

    def add_transition(
        self, word: str, state_in_id: int, state_out_id: int, tid: int, probabilities={}
    ):
        transition = SimpleTransition(
            self.states[state_out_id], word, tid, probabilities
        )
        self.states[state_in_id].add_transition(transition)

    def execute(self, doc: str):
        """Parse the given doc to return variable tokens

        Parameters
        ----------
        docs : str
            The document to parse

        Returns
        -------
        (str, {str: str})
            A tuple containing the parsed sentence (tokens separated by spaces) and a dict that map variables name to variable value
            found in the doc.
        """
        doc = doc.lower()
        doc = " ".join(doc.split("/"))
        words = doc.split(" ")
        words.append("<stop>")
        active_states = set(
            [ActiveState(self.start_state, (), history=tuple([0] * self.order))]
        )
        for word in words:
            if len(active_states) == 0:
                return doc, {}
            new_active_states = set()
            for state in active_states:
                new_active_states |= state.next_states(word)
            active_states = new_active_states
        shortest = None
        variables = None
        probability = 0
        for state in active_states:
            if state.state == self.stop_state:
                if (
                    shortest == None
                    or len(shortest) > len(state.captured_words)
                    or (
                        len(shortest) == len(state.captured_words)
                        and len(variables) < len(state.variables)
                    )
                    or (
                        len(shortest) == len(state.captured_words)
                        and len(variables) == len(state.variables)
                        and state.probability > probability
                    )
                ):
                    shortest = state.captured_words
                    variables = state.variables
                    probability = state.probability
        if shortest == None:
            return doc, {}
        return (" ".join(shortest), variables)

    def __repr__(self):
        rep = ""
        for state in self.states.values():
            for transition in state.next:
                rep += "{} --{}--> {}\n".format(
                    state.id, transition.word, transition.state_out.id
                )
        return rep[:-1]
