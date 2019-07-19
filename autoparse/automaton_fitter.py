import networkx as nx
import matplotlib.pyplot as plt

from autoparse.automaton import preprocess, Automaton


class Transition:
    def __init__(
        self,
        word: str,
        state_in,
        state_out,
        transition_ids=[],
        weight: int = 1,
        variables={},
    ):
        self.word = word
        self.state_in = state_in
        self.state_out = state_out
        self.weight = weight
        self.variables = variables
        self.transitions_ids = set(transition_ids)
        self.tid = next(iter(self.transitions_ids))
        self.p = {}

    def make_generic(self):
        generic = "*"
        best_count = 0
        for var, count in self.variables.items():
            if count > best_count:
                generic = "<$" + var + ">"
                best_count = count
        self.word = generic
        return generic

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.word == other.word
            and self.state_in == other.state_in
            and self.state_out == other.state_out
        )

    def __hash__(self):
        return hash(str(self.state_in) + str(self.state_out))

    def __repr__(self):
        return " {:6d} --{:^20}--> {:6d} ".format(
            self.state_in.id, self.word, self.state_out.id
        )


class TransitionSet:
    """A set implementation that add weights when adding a transition multiple times"""

    def __init__(self):
        self._dict = {}

    def __contains__(self, item):
        return item in self._dict

    def __iter__(self):
        return self._dict.keys().__iter__()

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return self._dict.__repr__()

    def _add(self, item):
        """Do not cumulate weight"""
        self._dict[item] = item

    def add(self, item):
        if not item in self._dict:
            self._dict[item] = item
        else:
            transition = self._dict[item]
            transition.weight += item.weight
            transition.transitions_ids |= item.transitions_ids
            for var in item.variables:
                if not var in transition.variables:
                    transition.variables[var] = 0
                transition.variables[var] += item.variables[var]

    def remove(self, item):
        if item in self._dict:
            del self._dict[item]


class State:
    def __init__(self, node_id: int, word: str):
        self.id = node_id
        self.transitions_in = TransitionSet()
        self.transitions_out = TransitionSet()
        self.word = word

    @property
    def weight(self):
        total_weight = 0
        for t in self.transitions_in:
            total_weight += t.weight
        return total_weight

    @property
    def child(self):
        for t in self.transitions_out:
            yield t.state_out

    @property
    def parents(self):
        for t in self.transitions_in:
            yield t.state_in

    def merge_on(self, state):
        transitions_to_delete = []
        for t in self.transitions_in:
            new_transition = Transition(
                t.word,
                t.state_in,
                state,
                transition_ids=t.transitions_ids,
                weight=t.weight,
                variables=t.variables,
            )
            state.add_transition_in(new_transition)
            transitions_to_delete.append(t)
        for t in self.transitions_out:
            new_transition = Transition(
                t.word,
                state,
                t.state_out,
                transition_ids=t.transitions_ids,
                weight=t.weight,
                variables=t.variables,
            )
            state.add_transition_out(new_transition)
            transitions_to_delete.append(t)
        for t in transitions_to_delete:
            t.state_out.remove_transition_in(t)

    def generify(self, limit_weight):
        if self.weight <= limit_weight:
            self.word = "*"
            for t in self.transitions_in:
                generic = t.make_generic()
                if generic != "*":
                    self.word = generic

    def get_generic_ancestors(self):
        """return the last ancestors connected by generics transion and drop those transitions"""
        if self.id == 0 or not self.word == "*":
            return [self], []
        else:
            ancestors = []
            intermediary_states = [self]
            for transition in self.transitions_in:
                new_ancestors, new_intermediary_states = (
                    transition.state_in.get_generic_ancestors()
                )
                ancestors += new_ancestors
                intermediary_states += new_intermediary_states
        return ancestors, intermediary_states

    def merge_generic_parents(self):
        if self.id == 0 or not self.word == "*":
            return
        ancestors, intermediary_states = self.get_generic_ancestors()
        transitions_ids = set()
        for state in intermediary_states:
            transitions_to_remove = list(state.transitions_in)
            for transition in transitions_to_remove:
                transitions_ids |= transition.transitions_ids
                state.remove_transition_in(transition)
        for ancestor in ancestors:
            self.add_transition_in(
                Transition(self.word, ancestor, self, transitions_ids)
            )

    def get_trivial_group(self):
        if len(self.transitions_in) <= 1:
            return set()
        merge_group = set()
        for parent in self.parents:
            if len(parent.transitions_out) == 1:
                merge_group.add(parent.id)
        return merge_group

    def add_transition_in(self, transition):
        self.transitions_in.add(transition)
        transition.state_in.__add_transition_out(transition)

    def add_transition_out(self, transition):
        self.transitions_out.add(transition)
        transition.state_out.__add_transition_in(transition)

    def remove_transition_in(self, transition):
        self.transitions_in.remove(transition)
        transition.state_in.__remove_transition_out(transition)

    def remove_transition_out(self, transition):
        self.transitions_out.remove(transition)
        transition.state_out.__remove_transition_in(transition)

    def __add_transition_in(self, transition):
        self.transitions_in._add(transition)

    def __add_transition_out(self, transition):
        self.transitions_out._add(transition)

    def __remove_transition_in(self, transition):
        self.transitions_in.remove(transition)

    def __remove_transition_out(self, transition):
        self.transitions_out.remove(transition)


class AutomatonFitter:
    """A class that fit an automaton on a list of documents

    The documents are assumed to be produced by a few numbers of templates that includes both
    fixed and variable words, produced with str.format() for instance. The fitted automaton 
    will guess which transitions hold variables and can extract them from new documents.

    Methods
    -------
    fit: 
        Fit the automaton

    build: 
        Return an executable automaton, should be called after fit 

    pprint:
        Pretty printer using Networkx and matplotlib

    print:
        Regular printer in string format
    """

    def __init__(self, docs, variables={}, order: int = 3):
        """Initialize the automaton

        Parameters
        ----------
        docs : str[]
            Documents to fit the automaton on
        variables: {str: str[]}
            keys are the name of variables (e.g. city) an values list of examples (e.g. ["Paris", "London", ...])
        order: int
            The memory size of the internal markov model used to predict path probability. 
        """
        self.nb_docs = len(docs)
        self.start_state = State(0, "<start>")
        self.stop_state = State(1, "<stop>")
        self.states = {0: self.start_state, 1: self.stop_state}
        self.stateCounter = 2
        self.transitionCounter = 1
        self.transitions_sequences = []
        self.order = order

        for var in variables.keys():
            variables[var] = set([v.lower() for v in variables[var]])

        for doc in docs:
            transition_sequence = []
            previous = self.stop_state
            doc = preprocess(doc)
            doc = " ".join(doc.split("/"))
            for word in doc.split(" ")[::-1]:
                state = self.create_state(word)
                var_count = self.get_variables(previous.word, variables)
                transition_out, tid = self.create_transition(state, previous, var_count)
                transition_sequence.append(tid)
                state.add_transition_out(transition_out)
                self.states[state.id] = state
                previous = state
            transition_out, tid = self.create_transition(self.start_state, state, {})
            transition_sequence.append(tid)
            self.start_state.add_transition_out(transition_out)
            transition_sequence = (transition_sequence + [0] * order)[::-1]
            self.transitions_sequences.append(transition_sequence)

    @staticmethod
    def get_variables(word, variables):
        """
        Return the list of variables this word is matching based on examples
        word: string
        variables: {string: set()}
        
        return: {string: int}
        """
        var_count = {}
        for var, examples in variables.items():
            if word in examples:
                var_count[var] = 1
        return var_count

    def create_transition(self, state_in, state_out, variables_count):
        tid = self.transitionCounter
        new_transition = Transition(
            state_out.word, state_in, state_out, [tid], variables=variables_count
        )
        self.transitionCounter += 1
        return new_transition, tid

    def create_state(self, word):
        new_state = State(self.stateCounter, word)
        self.stateCounter += 1
        return new_state

    def iterate_states(self, f, acc=None):
        """Apply `acc = f(state, acc)` on each state, return acc"""
        done = set()
        stack = [self.stop_state]
        while len(stack) > 0:
            state = stack.pop()
            if state.id in done:
                continue
            done.add(state.id)
            acc = f(state, acc)
            stack.extend(state.parents)
        return acc

    def count_word(self):
        def add_word(state, word_count):
            if not state.word in word_count:
                word_count[state.word] = 0
            word_count[state.word] += 1
            return word_count

        return self.iterate_states(add_word, {})

    def count_variables(self):
        def add_vars(state, vars_count):
            for t in state.transitions_in:
                for var, count in t.variables.items():
                    var = "<$" + var + ">"
                    if not var in vars_count:
                        vars_count[var] = 0
                    vars_count[var] += count
            return vars_count

        return self.iterate_states(add_vars, {})

    def make_state_generic(self, threshold: float = 0):
        limit_weight = threshold * self.nb_docs

        def generify(state, limit_weight):
            state.generify(limit_weight)
            return limit_weight

        self.iterate_states(generify, limit_weight)

    def simplify_generic_chains(self):
        def merge_generics(state, acc):
            state.merge_generic_parents()
            return acc

        self.iterate_states(merge_generics)

    def merge_trivial_groups(self):
        def trivial_group(state, group_list):
            group_list.append(state.get_trivial_group())
            return group_list

        merge_group_list = self.iterate_states(trivial_group, [])
        for group in merge_group_list:
            self.merge_group(group, 0)

    def remove_rare_transitions(self, freq: float):
        limit_weight = freq * self.nb_docs

        def remove_rare_out_transitions(state, limit_weight):
            transitions_to_remove = []
            for t in state.transitions_out:
                if t.weight <= limit_weight:
                    transitions_to_remove.append(t)
            for t in transitions_to_remove:
                state.remove_transition_out(t)
            return limit_weight

        self.iterate_states(remove_rare_out_transitions, limit_weight)

    def merge_group(self, merge_group, threshold):
        if (
            not len(merge_group) >= 2
            or not len(merge_group) >= threshold * self.nb_docs
        ):
            return False
        merge_state = self.states[next(iter(merge_group))]
        merge_group.remove(merge_state.id)

        def merge(state, acc):
            if state.id in merge_group:
                state.merge_on(merge_state)
            return acc

        self.iterate_states(merge)
        return True

    def find_merge_group(self, word: str):
        incompatibles = set()
        merge_group = set()
        stack = [(self.stop_state, set())]  # (state, set of descendants)
        visited = {}  # state -> [nb_visit, set of descendants]
        while len(stack) > 0:
            state, descendants = stack.pop()
            new_descendant = set()
            if state.word == word:
                new_descendant.add(state.id)
                merge_group.add(state.id)
                for descendant_id in descendants:
                    incompatibles.add((descendant_id, state.id))
                    incompatibles.add((state.id, descendant_id))
            if not state in visited:
                visited[state] = [0, set()]
            visited[state][0] += 1
            visited[state][1] |= descendants
            visited[state][1] |= new_descendant
            if visited[state][0] >= len(state.transitions_out):
                descendants = visited[state][1]
                for parent in state.parents:
                    stack.append((parent, descendants))
        return self.remove_incompatibles(merge_group, incompatibles)

    def remove_incompatibles(self, merge_group, incompatibles):
        incompatible_count = {}
        for state1, state2 in incompatibles:
            if not state1 in incompatible_count:
                incompatible_count[state1] = 0
            if not state2 in incompatible_count:
                incompatible_count[state2] = 0
            incompatible_count[state1] += 1
            incompatible_count[state2] += 1
        for state1, state2 in incompatibles:
            if state1 in merge_group and state2 in merge_group:
                if incompatible_count[state1] > incompatible_count[state2]:
                    merge_group.remove(state1)
                else:
                    merge_group.remove(state2)
        return merge_group

    def merge_word(self, word: str, threshold: float = 0):
        return self.merge_group(self.find_merge_group(word), threshold)

    def reduce(self, threshold: float = 0, variables: bool = False, word_black_list=[]):
        """
        Merge either on words or on variables. Should merge on variable only after 
        `self.make_state_generic` has been called.
        """
        count_function = self.count_word
        if variables == True:
            count_function = self.count_variables
        done = False
        black_list = set([w.lower() for w in word_black_list])
        for word, nb_occurrences in self.count_word().items():
            if nb_occurrences < threshold * self.nb_docs:
                black_list.add(word)
        while not done:
            transition_count = [
                (word, nb_occurrences)
                for word, nb_occurrences in count_function().items()
                if word not in black_list
            ]
            if len(transition_count) == 0:
                done = True
                break
            transition_count.sort(key=lambda x: x[1])
            word, count = transition_count.pop()
            if count > 1:
                success = self.merge_word(word, threshold)
                if not success:
                    black_list.add(word)
            else:
                done = True

    def compute_transition_probability(self):
        for transitions_sequence in self.transitions_sequences:
            previous_transition = None
            state = self.start_state
            for i in range(self.order, len(transitions_sequence)):
                tid = transitions_sequence[i]
                history = tuple(transitions_sequence[i - self.order : i])
                found = False
                for transition in state.transitions_out:
                    if tid in transition.transitions_ids:
                        found = True
                        transitions_sequence[i] = transition.tid
                        previous_transition = transition
                        state = transition.state_out
                        if not history in transition.p:
                            transition.p[history] = 0
                        transition.p[history] += 1
                        break
                if not found and previous_transition != None:
                    if tid in previous_transition.transitions_ids:
                        found = True
                        transitions_sequence[i] = previous_transition.tid
                        if not history in previous_transition.p:
                            previous_transition.p[history] = 0
                        previous_transition.p[history] += 1
                if not found:
                    break

        def normalize_probabilities(state, acc):
            for transition in state.transitions_in:
                total = 0
                for history, count in transition.p.items():
                    total += count
                for history in transition.p.keys():
                    transition.p[history] /= total
            return acc

        self.iterate_states(normalize_probabilities)

    def build(self):
        """Build and return an executable and lightweight automaton
        """
        self.compute_transition_probability()

        stationary_transition_id = self.transitionCounter
        self.transitionCounter += 1

        def build_state(state, automaton):
            automaton.add_state(state.id)
            for transition in state.transitions_in:
                automaton.add_state(transition.state_in.id)
                automaton.add_transition(
                    transition.word,
                    transition.state_in.id,
                    state.id,
                    transition.tid,
                    transition.p,
                )
                if transition.word == "*":
                    automaton.add_transition(
                        transition.word,
                        state.id,
                        state.id,
                        stationary_transition_id,
                        transition.p,
                    )
            return automaton

        return self.iterate_states(build_state, Automaton(self.order))

    def fit(self, threshold: float = 0.2, min_freq: float = 0, word_black_list=[]):
        """Fit the automaton
        
        Parameters
        ----------
        threshold : float
            The frequency threshold, each pattern should have a frequency higher than this threshold
        min_freq: float
            The minimum frequency, every transition with lower frequency will be discarded. Set 0
            to keep all transitions.
        word_black_list: str[]
            Initialize the blacklist of words. Words with frequency higher than the threshold but that 
            are not part of the hidden template should be added to the blacklist if known.
        """
        self.reduce(threshold, word_black_list=word_black_list)
        self.make_state_generic(threshold)
        self.reduce(threshold, variables=True)
        self.simplify_generic_chains()
        self.merge_trivial_groups()
        if min_freq > 0:
            self.remove_rare_transitions(min_freq)

    def fit_build(self, threshold: float = 0.2, min_freq: float = 0, word_black_list=[]):
        """Fit and return an executable automaton
        
        Parameters
        ----------
        threshold : float
            The frequency threshold, each pattern should have a frequency higher than this threshold
        min_freq: float
            The minimum frequency, every transition with lower frequency will be discarded. Set 0
            to keep all transitions.
        word_black_list: str[]
            Initialize the blacklist of words. Words with frequency higher than the threshold but that 
            are not part of the hidden template should be added to the blacklist if known.
        """
        self.fit(threshold, min_freq, word_black_list)
        return self.build()

    def graph(self):
        """Return a networkx graph object that correspond to the automaton
        """
        G = nx.DiGraph()
        done = set()
        stack = [self.stop_state]
        while len(stack) > 0:
            state = stack.pop()
            done.add(state.id)
            for t in state.transitions_in:
                G.add_edge(
                    t.state_in.id, t.state_out.id, label=t.word + " - " + str(t.weight)
                )
                if not t.state_in.id in done:
                    stack.append(t.state_in)
        return G

    def pprint(self):
        """Plot a graphic representation of the automaton
        """
        G = self.graph()
        fig = plt.figure(figsize=(14, 12))
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, with_labels=True, alpha=0.6)
        labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    def print(self):
        """Print the transitions in string format
        """

        def print_transitions(state, acc):
            for t in state.transitions_in:
                print(t)
            return acc

        self.iterate_states(print_transitions)
