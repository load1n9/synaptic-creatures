let connections2 = 0;
class Connection {
    from;
    to;
    ID = Connection.uid();
    gain = 1;
    gater = null;
    weight;
    constructor(from1, to1, weight1){
        this.from = from1;
        this.to = to1;
        this.weight = weight1 || Math.random() * 0.2 - 0.1;
    }
    static uid() {
        return connections2++;
    }
}
let neurons = 0;
const squash1 = {
    LOGISTIC: (x, derivate)=>{
        let fx = 1 / (1 + Math.exp(-x));
        if (!derivate) return fx;
        return fx * (1 - fx);
    },
    TANH: (x, derivate)=>{
        if (derivate) return 1 - Math.pow(Math.tanh(x), 2);
        return Math.tanh(x);
    },
    IDENTITY: (x, derivate)=>{
        return derivate ? 1 : x;
    },
    HLIM: (x, derivate)=>{
        return derivate ? 1 : x > 0 ? 1 : 0;
    },
    RELU: (x, derivate)=>{
        if (derivate) return x > 0 ? 1 : 0;
        return x > 0 ? x : 0;
    }
};
class Neuron {
    static squash = squash1;
    constructor(){
        this.ID = Neuron.uid();
        this.connections = {
            inputs: {
            },
            projected: {
            },
            gated: {
            }
        };
        this.error = {
            responsibility: 0,
            projected: 0,
            gated: 0
        };
        this.trace = {
            elegibility: {
            },
            extended: {
            },
            influences: {
            }
        };
        this.state = 0;
        this.old = 0;
        this.activation = 0;
        this.selfconnection = new Connection(this, this, 0);
        this.squash = Neuron.squash.LOGISTIC;
        this.neighboors = {
        };
        this.bias = Math.random() * 0.2 - 0.1;
    }
    activate(input) {
        if (typeof input != 'undefined') {
            this.activation = input;
            this.derivative = 0;
            this.bias = 0;
            return this.activation;
        }
        this.old = this.state;
        this.state = this.selfconnection.gain * this.selfconnection.weight * this.state + this.bias;
        for(var i in this.connections.inputs){
            var input = this.connections.inputs[i];
            this.state += input.from.activation * input.weight * input.gain;
        }
        this.activation = this.squash(this.state);
        this.derivative = this.squash(this.state, true);
        var influences = [];
        for(var id in this.trace.extended){
            var neuron = this.neighboors[id];
            var influence = neuron.selfconnection.gater == this ? neuron.old : 0;
            for(var incoming in this.trace.influences[neuron.ID]){
                influence += this.trace.influences[neuron.ID][incoming].weight * this.trace.influences[neuron.ID][incoming].from.activation;
            }
            influences[neuron.ID] = influence;
        }
        for(var i in this.connections.inputs){
            var input = this.connections.inputs[i];
            this.trace.elegibility[input.ID] = this.selfconnection.gain * this.selfconnection.weight * this.trace.elegibility[input.ID] + input.gain * input.from.activation;
            for(var id in this.trace.extended){
                var xtrace = this.trace.extended[id];
                var neuron = this.neighboors[id];
                var influence = influences[neuron.ID];
                xtrace[input.ID] = neuron.selfconnection.gain * neuron.selfconnection.weight * xtrace[input.ID] + this.derivative * this.trace.elegibility[input.ID] * influence;
            }
        }
        for(var connection in this.connections.gated){
            this.connections.gated[connection].gain = this.activation;
        }
        return this.activation;
    }
    propagate(rate, target) {
        var error = 0;
        var isOutput = typeof target != 'undefined';
        if (isOutput) this.error.responsibility = this.error.projected = target - this.activation;
        else {
            for(var id in this.connections.projected){
                var connection = this.connections.projected[id];
                var neuron = connection.to;
                error += neuron.error.responsibility * connection.gain * connection.weight;
            }
            this.error.projected = this.derivative * error;
            error = 0;
            for(var id in this.trace.extended){
                var neuron = this.neighboors[id];
                var influence = neuron.selfconnection.gater == this ? neuron.old : 0;
                for(var input in this.trace.influences[id]){
                    influence += this.trace.influences[id][input].weight * this.trace.influences[neuron.ID][input].from.activation;
                }
                error += neuron.error.responsibility * influence;
            }
            this.error.gated = this.derivative * error;
            this.error.responsibility = this.error.projected + this.error.gated;
        }
        rate = rate || 0.1;
        for(var id in this.connections.inputs){
            var input = this.connections.inputs[id];
            var gradient = this.error.projected * this.trace.elegibility[input.ID];
            for(var id in this.trace.extended){
                var neuron = this.neighboors[id];
                gradient += neuron.error.responsibility * this.trace.extended[neuron.ID][input.ID];
            }
            input.weight += rate * gradient;
        }
        this.bias += rate * this.error.responsibility;
    }
    project(neuron, weight) {
        if (neuron == this) {
            this.selfconnection.weight = 1;
            return this.selfconnection;
        }
        var connected = this.connected(neuron);
        if (connected && connected.type == 'projected') {
            if (typeof weight != 'undefined') connected.connection.weight = weight;
            return connected.connection;
        } else {
            var connection = new Connection(this, neuron, weight);
        }
        this.connections.projected[connection.ID] = connection;
        this.neighboors[neuron.ID] = neuron;
        neuron.connections.inputs[connection.ID] = connection;
        neuron.trace.elegibility[connection.ID] = 0;
        for(var id in neuron.trace.extended){
            var trace = neuron.trace.extended[id];
            trace[connection.ID] = 0;
        }
        return connection;
    }
    gate(connection) {
        this.connections.gated[connection.ID] = connection;
        var neuron = connection.to;
        if (!(neuron.ID in this.trace.extended)) {
            this.neighboors[neuron.ID] = neuron;
            var xtrace = this.trace.extended[neuron.ID] = {
            };
            for(var id in this.connections.inputs){
                var input = this.connections.inputs[id];
                xtrace[input.ID] = 0;
            }
        }
        if (neuron.ID in this.trace.influences) this.trace.influences[neuron.ID].push(connection);
        else this.trace.influences[neuron.ID] = [
            connection
        ];
        connection.gater = this;
    }
    selfconnected() {
        return this.selfconnection.weight !== 0;
    }
    connected(neuron) {
        var result = {
            type: null,
            connection: false
        };
        if (this == neuron) {
            if (this.selfconnected()) {
                result.type = 'selfconnection';
                result.connection = this.selfconnection;
                return result;
            } else return false;
        }
        for(var type in this.connections){
            for(var connection in this.connections[type]){
                var connection = this.connections[type][connection];
                if (connection.to == neuron) {
                    result.type = type;
                    result.connection = connection;
                    return result;
                } else if (connection.from == neuron) {
                    result.type = type;
                    result.connection = connection;
                    return result;
                }
            }
        }
        return false;
    }
    clear() {
        for(var trace in this.trace.elegibility){
            this.trace.elegibility[trace] = 0;
        }
        for(var trace in this.trace.extended){
            for(var extended in this.trace.extended[trace]){
                this.trace.extended[trace][extended] = 0;
            }
        }
        this.error.responsibility = this.error.projected = this.error.gated = 0;
    }
    reset() {
        this.clear();
        for(var type in this.connections){
            for(var connection in this.connections[type]){
                this.connections[type][connection].weight = Math.random() * 0.2 - 0.1;
            }
        }
        this.bias = Math.random() * 0.2 - 0.1;
        this.old = this.state = this.activation = 0;
    }
    optimize(optimized, layer) {
        optimized = optimized || {
        };
        var store_activation = [];
        var store_trace = [];
        var store_propagation = [];
        var varID = optimized.memory || 0;
        var neurons1 = optimized.neurons || 1;
        var inputs = optimized.inputs || [];
        var targets = optimized.targets || [];
        var outputs = optimized.outputs || [];
        var variables = optimized.variables || {
        };
        var activation_sentences = optimized.activation_sentences || [];
        var trace_sentences = optimized.trace_sentences || [];
        var propagation_sentences = optimized.propagation_sentences || [];
        var layers = optimized.layers || {
            __count: 0,
            __neuron: 0
        };
        var allocate = function(store) {
            var allocated = layer in layers && store[layers.__count];
            if (!allocated) {
                layers.__count = store.push([]) - 1;
                layers[layer] = layers.__count;
            }
        };
        allocate(activation_sentences);
        allocate(trace_sentences);
        allocate(propagation_sentences);
        var currentLayer = layers.__count;
        var getVar = function() {
            var args = Array.prototype.slice.call(arguments);
            if (args.length == 1) {
                if (args[0] == 'target') {
                    var id = 'target_' + targets.length;
                    targets.push(varID);
                } else var id = args[0];
                if (id in variables) return variables[id];
                return variables[id] = {
                    value: 0,
                    id: varID++
                };
            } else {
                var extended = args.length > 2;
                if (extended) var value = args.pop();
                var unit = args.shift();
                var prop = args.pop();
                if (!extended) var value = unit[prop];
                var id = prop + '_';
                for(var i = 0; i < args.length; i++)id += args[i] + '_';
                id += unit.ID;
                if (id in variables) return variables[id];
                return variables[id] = {
                    value: value,
                    id: varID++
                };
            }
        };
        var buildSentence = function() {
            var args = Array.prototype.slice.call(arguments);
            var store = args.pop();
            var sentence = '';
            for(var i = 0; i < args.length; i++)if (typeof args[i] == 'string') sentence += args[i];
            else sentence += 'F[' + args[i].id + ']';
            store.push(sentence + ';');
        };
        var isEmpty = function(obj) {
            for(var prop in obj){
                if (obj.hasOwnProperty(prop)) return false;
            }
            return true;
        };
        var noProjections = isEmpty(this.connections.projected);
        var noGates = isEmpty(this.connections.gated);
        var isInput = layer == 'input' ? true : isEmpty(this.connections.inputs);
        var isOutput = layer == 'output' ? true : noProjections && noGates;
        var rate = getVar('rate');
        var activation = getVar(this, 'activation');
        if (isInput) inputs.push(activation.id);
        else {
            activation_sentences[currentLayer].push(store_activation);
            trace_sentences[currentLayer].push(store_trace);
            propagation_sentences[currentLayer].push(store_propagation);
            var old = getVar(this, 'old');
            var state = getVar(this, 'state');
            var bias = getVar(this, 'bias');
            if (this.selfconnection.gater) var self_gain = getVar(this.selfconnection, 'gain');
            if (this.selfconnected()) var self_weight = getVar(this.selfconnection, 'weight');
            buildSentence(old, ' = ', state, store_activation);
            if (this.selfconnected()) {
                if (this.selfconnection.gater) buildSentence(state, ' = ', self_gain, ' * ', self_weight, ' * ', state, ' + ', bias, store_activation);
                else buildSentence(state, ' = ', self_weight, ' * ', state, ' + ', bias, store_activation);
            } else buildSentence(state, ' = ', bias, store_activation);
            for(var i in this.connections.inputs){
                var input = this.connections.inputs[i];
                var input_activation = getVar(input.from, 'activation');
                var input_weight = getVar(input, 'weight');
                if (input.gater) var input_gain = getVar(input, 'gain');
                if (this.connections.inputs[i].gater) buildSentence(state, ' += ', input_activation, ' * ', input_weight, ' * ', input_gain, store_activation);
                else buildSentence(state, ' += ', input_activation, ' * ', input_weight, store_activation);
            }
            var derivative = getVar(this, 'derivative');
            switch(this.squash){
                case Neuron.squash.LOGISTIC:
                    buildSentence(activation, ' = (1 / (1 + Math.exp(-', state, ')))', store_activation);
                    buildSentence(derivative, ' = ', activation, ' * (1 - ', activation, ')', store_activation);
                    break;
                case Neuron.squash.TANH:
                    var eP = getVar('aux');
                    var eN = getVar('aux_2');
                    buildSentence(eP, ' = Math.exp(', state, ')', store_activation);
                    buildSentence(eN, ' = 1 / ', eP, store_activation);
                    buildSentence(activation, ' = (', eP, ' - ', eN, ') / (', eP, ' + ', eN, ')', store_activation);
                    buildSentence(derivative, ' = 1 - (', activation, ' * ', activation, ')', store_activation);
                    break;
                case Neuron.squash.IDENTITY:
                    buildSentence(activation, ' = ', state, store_activation);
                    buildSentence(derivative, ' = 1', store_activation);
                    break;
                case Neuron.squash.HLIM:
                    buildSentence(activation, ' = +(', state, ' > 0)', store_activation);
                    buildSentence(derivative, ' = 1', store_activation);
                    break;
                case Neuron.squash.RELU:
                    buildSentence(activation, ' = ', state, ' > 0 ? ', state, ' : 0', store_activation);
                    buildSentence(derivative, ' = ', state, ' > 0 ? 1 : 0', store_activation);
                    break;
            }
            for(var id in this.trace.extended){
                var neuron = this.neighboors[id];
                var influence = getVar('influences[' + neuron.ID + ']');
                var neuron_old = getVar(neuron, 'old');
                var initialized = false;
                if (neuron.selfconnection.gater == this) {
                    buildSentence(influence, ' = ', neuron_old, store_trace);
                    initialized = true;
                }
                for(var incoming in this.trace.influences[neuron.ID]){
                    var incoming_weight = getVar(this.trace.influences[neuron.ID][incoming], 'weight');
                    var incoming_activation = getVar(this.trace.influences[neuron.ID][incoming].from, 'activation');
                    if (initialized) buildSentence(influence, ' += ', incoming_weight, ' * ', incoming_activation, store_trace);
                    else {
                        buildSentence(influence, ' = ', incoming_weight, ' * ', incoming_activation, store_trace);
                        initialized = true;
                    }
                }
            }
            for(var i in this.connections.inputs){
                var input = this.connections.inputs[i];
                if (input.gater) var input_gain = getVar(input, 'gain');
                var input_activation = getVar(input.from, 'activation');
                var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
                if (this.selfconnected()) {
                    if (this.selfconnection.gater) {
                        if (input.gater) buildSentence(trace, ' = ', self_gain, ' * ', self_weight, ' * ', trace, ' + ', input_gain, ' * ', input_activation, store_trace);
                        else buildSentence(trace, ' = ', self_gain, ' * ', self_weight, ' * ', trace, ' + ', input_activation, store_trace);
                    } else {
                        if (input.gater) buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ', input_gain, ' * ', input_activation, store_trace);
                        else buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ', input_activation, store_trace);
                    }
                } else {
                    if (input.gater) buildSentence(trace, ' = ', input_gain, ' * ', input_activation, store_trace);
                    else buildSentence(trace, ' = ', input_activation, store_trace);
                }
                for(var id in this.trace.extended){
                    var neuron = this.neighboors[id];
                    var influence = getVar('influences[' + neuron.ID + ']');
                    var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
                    var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID, this.trace.extended[neuron.ID][input.ID]);
                    if (neuron.selfconnected()) var neuron_self_weight = getVar(neuron.selfconnection, 'weight');
                    if (neuron.selfconnection.gater) var neuron_self_gain = getVar(neuron.selfconnection, 'gain');
                    if (neuron.selfconnected()) {
                        if (neuron.selfconnection.gater) buildSentence(xtrace, ' = ', neuron_self_gain, ' * ', neuron_self_weight, ' * ', xtrace, ' + ', derivative, ' * ', trace, ' * ', influence, store_trace);
                        else buildSentence(xtrace, ' = ', neuron_self_weight, ' * ', xtrace, ' + ', derivative, ' * ', trace, ' * ', influence, store_trace);
                    } else buildSentence(xtrace, ' = ', derivative, ' * ', trace, ' * ', influence, store_trace);
                }
            }
            for(var connection in this.connections.gated){
                var gated_gain = getVar(this.connections.gated[connection], 'gain');
                buildSentence(gated_gain, ' = ', activation, store_activation);
            }
        }
        if (!isInput) {
            var responsibility = getVar(this, 'error', 'responsibility', this.error.responsibility);
            if (isOutput) {
                var target = getVar('target');
                buildSentence(responsibility, ' = ', target, ' - ', activation, store_propagation);
                for(var id in this.connections.inputs){
                    var input = this.connections.inputs[id];
                    var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
                    var input_weight = getVar(input, 'weight');
                    buildSentence(input_weight, ' += ', rate, ' * (', responsibility, ' * ', trace, ')', store_propagation);
                }
                outputs.push(activation.id);
            } else {
                if (!noProjections && !noGates) {
                    var error = getVar('aux');
                    for(var id in this.connections.projected){
                        var connection = this.connections.projected[id];
                        var neuron = connection.to;
                        var connection_weight = getVar(connection, 'weight');
                        var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                        if (connection.gater) {
                            var connection_gain = getVar(connection, 'gain');
                            buildSentence(error, ' += ', neuron_responsibility, ' * ', connection_gain, ' * ', connection_weight, store_propagation);
                        } else buildSentence(error, ' += ', neuron_responsibility, ' * ', connection_weight, store_propagation);
                    }
                    var projected = getVar(this, 'error', 'projected', this.error.projected);
                    buildSentence(projected, ' = ', derivative, ' * ', error, store_propagation);
                    buildSentence(error, ' = 0', store_propagation);
                    for(var id in this.trace.extended){
                        var neuron = this.neighboors[id];
                        var influence = getVar('aux_2');
                        var neuron_old = getVar(neuron, 'old');
                        if (neuron.selfconnection.gater == this) buildSentence(influence, ' = ', neuron_old, store_propagation);
                        else buildSentence(influence, ' = 0', store_propagation);
                        for(var input in this.trace.influences[neuron.ID]){
                            var connection = this.trace.influences[neuron.ID][input];
                            var connection_weight = getVar(connection, 'weight');
                            var neuron_activation = getVar(connection.from, 'activation');
                            buildSentence(influence, ' += ', connection_weight, ' * ', neuron_activation, store_propagation);
                        }
                        var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                        buildSentence(error, ' += ', neuron_responsibility, ' * ', influence, store_propagation);
                    }
                    var gated = getVar(this, 'error', 'gated', this.error.gated);
                    buildSentence(gated, ' = ', derivative, ' * ', error, store_propagation);
                    buildSentence(responsibility, ' = ', projected, ' + ', gated, store_propagation);
                    for(var id in this.connections.inputs){
                        var input = this.connections.inputs[id];
                        var gradient = getVar('aux');
                        var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
                        buildSentence(gradient, ' = ', projected, ' * ', trace, store_propagation);
                        for(var id in this.trace.extended){
                            var neuron = this.neighboors[id];
                            var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                            var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID, this.trace.extended[neuron.ID][input.ID]);
                            buildSentence(gradient, ' += ', neuron_responsibility, ' * ', xtrace, store_propagation);
                        }
                        var input_weight = getVar(input, 'weight');
                        buildSentence(input_weight, ' += ', rate, ' * ', gradient, store_propagation);
                    }
                } else if (noGates) {
                    buildSentence(responsibility, ' = 0', store_propagation);
                    for(var id in this.connections.projected){
                        var connection = this.connections.projected[id];
                        var neuron = connection.to;
                        var connection_weight = getVar(connection, 'weight');
                        var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                        if (connection.gater) {
                            var connection_gain = getVar(connection, 'gain');
                            buildSentence(responsibility, ' += ', neuron_responsibility, ' * ', connection_gain, ' * ', connection_weight, store_propagation);
                        } else buildSentence(responsibility, ' += ', neuron_responsibility, ' * ', connection_weight, store_propagation);
                    }
                    buildSentence(responsibility, ' *= ', derivative, store_propagation);
                    for(var id in this.connections.inputs){
                        var input = this.connections.inputs[id];
                        var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
                        var input_weight = getVar(input, 'weight');
                        buildSentence(input_weight, ' += ', rate, ' * (', responsibility, ' * ', trace, ')', store_propagation);
                    }
                } else if (noProjections) {
                    buildSentence(responsibility, ' = 0', store_propagation);
                    for(var id in this.trace.extended){
                        var neuron = this.neighboors[id];
                        var influence = getVar('aux');
                        var neuron_old = getVar(neuron, 'old');
                        if (neuron.selfconnection.gater == this) buildSentence(influence, ' = ', neuron_old, store_propagation);
                        else buildSentence(influence, ' = 0', store_propagation);
                        for(var input in this.trace.influences[neuron.ID]){
                            var connection = this.trace.influences[neuron.ID][input];
                            var connection_weight = getVar(connection, 'weight');
                            var neuron_activation = getVar(connection.from, 'activation');
                            buildSentence(influence, ' += ', connection_weight, ' * ', neuron_activation, store_propagation);
                        }
                        var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                        buildSentence(responsibility, ' += ', neuron_responsibility, ' * ', influence, store_propagation);
                    }
                    buildSentence(responsibility, ' *= ', derivative, store_propagation);
                    for(var id in this.connections.inputs){
                        var input = this.connections.inputs[id];
                        var gradient = getVar('aux');
                        buildSentence(gradient, ' = 0', store_propagation);
                        for(var id in this.trace.extended){
                            var neuron = this.neighboors[id];
                            var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                            var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID, this.trace.extended[neuron.ID][input.ID]);
                            buildSentence(gradient, ' += ', neuron_responsibility, ' * ', xtrace, store_propagation);
                        }
                        var input_weight = getVar(input, 'weight');
                        buildSentence(input_weight, ' += ', rate, ' * ', gradient, store_propagation);
                    }
                }
            }
            buildSentence(bias, ' += ', rate, ' * ', responsibility, store_propagation);
        }
        return {
            memory: varID,
            neurons: neurons1 + 1,
            inputs: inputs,
            outputs: outputs,
            targets: targets,
            variables: variables,
            activation_sentences: activation_sentences,
            trace_sentences: trace_sentences,
            propagation_sentences: propagation_sentences,
            layers: layers
        };
    }
    static uid() {
        return neurons++;
    }
    static quantity() {
        return {
            neurons: neurons,
            connections: connections2
        };
    }
}
const shuffleInplace = (o)=>{
    for(let j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
    return o;
};
const cost1 = {
    CROSS_ENTROPY: (target, output)=>{
        var crossentropy = 0;
        for(var i in output)crossentropy -= target[i] * Math.log(output[i] + 0.000000000000001) + (1 - target[i]) * Math.log(1 + 0.000000000000001 - output[i]);
        return crossentropy;
    },
    MSE: (target, output)=>{
        var mse = 0;
        for(var i = 0; i < output.length; i++)mse += Math.pow(target[i] - output[i], 2);
        return mse / output.length;
    },
    BINARY: (target, output)=>{
        var misses = 0;
        for(var i = 0; i < output.length; i++)misses += Math.round(target[i] * 2) != Math.round(output[i] * 2);
        return misses;
    }
};
class Trainer {
    static cost = cost1;
    constructor(network1, options1){
        options1 = options1 || {
        };
        this.network = network1;
        this.rate = options1.rate || 0.2;
        this.iterations = options1.iterations || 100000;
        this.error = options1.error || 0.005;
        this.cost = options1.cost || null;
        this.crossValidate = options1.crossValidate || null;
    }
    train(set, options) {
        var error = 1;
        var iterations = bucketSize = 0;
        var abort = false;
        var currentRate;
        var cost2 = options && options.cost || this.cost || Trainer.cost.MSE;
        var crossValidate = false, testSet, trainSet;
        var start = Date.now();
        if (options) {
            if (options.iterations) this.iterations = options.iterations;
            if (options.error) this.error = options.error;
            if (options.rate) this.rate = options.rate;
            if (options.cost) this.cost = options.cost;
            if (options.schedule) this.schedule = options.schedule;
            if (options.customLog) {
                console.log('Deprecated: use schedule instead of customLog');
                this.schedule = options.customLog;
            }
            if (this.crossValidate || options.crossValidate) {
                if (!this.crossValidate) this.crossValidate = {
                };
                crossValidate = true;
                if (options.crossValidate.testSize) this.crossValidate.testSize = options.crossValidate.testSize;
                if (options.crossValidate.testError) this.crossValidate.testError = options.crossValidate.testError;
            }
        }
        currentRate = this.rate;
        if (Array.isArray(this.rate)) {
            var bucketSize = Math.floor(this.iterations / this.rate.length);
        }
        if (crossValidate) {
            var numTrain = Math.ceil((1 - this.crossValidate.testSize) * set.length);
            trainSet = set.slice(0, numTrain);
            testSet = set.slice(numTrain);
        }
        var lastError = 0;
        while(!abort && iterations < this.iterations && error > this.error){
            if (crossValidate && error <= this.crossValidate.testError) {
                break;
            }
            var currentSetSize = set.length;
            error = 0;
            iterations++;
            if (bucketSize > 0) {
                var currentBucket = Math.floor(iterations / bucketSize);
                currentRate = this.rate[currentBucket] || currentRate;
            }
            if (typeof this.rate === 'function') {
                currentRate = this.rate(iterations, lastError);
            }
            if (crossValidate) {
                this._trainSet(trainSet, currentRate, cost2);
                error += this.test(testSet).error;
                currentSetSize = 1;
            } else {
                error += this._trainSet(set, currentRate, cost2);
                currentSetSize = set.length;
            }
            error /= currentSetSize;
            lastError = error;
            if (options) {
                if (this.schedule && this.schedule.every && iterations % this.schedule.every == 0) abort = this.schedule.do({
                    error: error,
                    iterations: iterations,
                    rate: currentRate
                });
                else if (options.log && iterations % options.log == 0) {
                    console.log('iterations', iterations, 'error', error, 'rate', currentRate);
                }
                if (options.shuffle) shuffleInplace(set);
            }
        }
        var results = {
            error: error,
            iterations: iterations,
            time: Date.now() - start
        };
        return results;
    }
    trainAsync(set, options) {
        var train = this.workerTrain.bind(this);
        return new Promise(function(resolve, reject) {
            try {
                train(set, resolve, options, true);
            } catch (e) {
                reject(e);
            }
        });
    }
    _trainSet(set, currentRate, costFunction) {
        var errorSum = 0;
        for(var i = 0; i < set.length; i++){
            var input = set[i].input;
            var target = set[i].output;
            var output = this.network.activate(input);
            this.network.propagate(currentRate, target);
            errorSum += costFunction(target, output);
        }
        return errorSum;
    }
    test(set, options) {
        var error = 0;
        var input, output, target;
        var cost3 = options && options.cost || this.cost || Trainer.cost.MSE;
        var start = Date.now();
        for(var i = 0; i < set.length; i++){
            input = set[i].input;
            target = set[i].output;
            output = this.network.activate(input);
            error += cost3(target, output);
        }
        error /= set.length;
        var results = {
            error: error,
            time: Date.now() - start
        };
        return results;
    }
    workerTrain(set, callback, options, suppressWarning) {
        if (!suppressWarning) {
            console.warn('Deprecated: do not use `workerTrain`, use `trainAsync` instead.');
        }
        var that = this;
        if (!this.network.optimized) this.network.optimize();
        var worker = this.network.worker(this.network.optimized.memory, set, options);
        worker.onmessage = function(e) {
            switch(e.data.action){
                case 'done':
                    var iterations = e.data.message.iterations;
                    var error = e.data.message.error;
                    var time = e.data.message.time;
                    that.network.optimized.ownership(e.data.memoryBuffer);
                    callback({
                        error: error,
                        iterations: iterations,
                        time: time
                    });
                    worker.terminate();
                    break;
                case 'log':
                    console.log(e.data.message);
                case 'schedule':
                    if (options && options.schedule && typeof options.schedule.do === 'function') {
                        var scheduled = options.schedule.do;
                        scheduled(e.data.message);
                    }
                    break;
            }
        };
        worker.postMessage({
            action: 'startTraining'
        });
    }
    XOR(options) {
        if (this.network.inputs() != 2 || this.network.outputs() != 1) throw new Error('Incompatible network (2 inputs, 1 output)');
        var defaults = {
            iterations: 100000,
            log: false,
            shuffle: true,
            cost: Trainer.cost.MSE
        };
        if (options) for(var i in options)defaults[i] = options[i];
        return this.train([
            {
                input: [
                    0,
                    0
                ],
                output: [
                    0
                ]
            },
            {
                input: [
                    1,
                    0
                ],
                output: [
                    1
                ]
            },
            {
                input: [
                    0,
                    1
                ],
                output: [
                    1
                ]
            },
            {
                input: [
                    1,
                    1
                ],
                output: [
                    0
                ]
            }
        ], defaults);
    }
    DSR(options) {
        options = options || {
        };
        var targets = options.targets || [
            2,
            4,
            7,
            8
        ];
        var distractors = options.distractors || [
            3,
            5,
            6,
            9
        ];
        var prompts = options.prompts || [
            0,
            1
        ];
        var length = options.length || 24;
        var criterion = options.success || 0.95;
        var iterations = options.iterations || 100000;
        var rate = options.rate || 0.1;
        var log = options.log || 0;
        var schedule = options.schedule || {
        };
        var cost4 = options.cost || this.cost || Trainer.cost.CROSS_ENTROPY;
        var trial, correct, i, j, success;
        trial = correct = i = j = success = 0;
        var error = 1, symbols = targets.length + distractors.length + prompts.length;
        var noRepeat = function(range, avoid) {
            var number = Math.random() * range | 0;
            var used = false;
            for(var i1 in avoid)if (number == avoid[i1]) used = true;
            return used ? noRepeat(range, avoid) : number;
        };
        var equal = function(prediction, output) {
            for(var i1 in prediction)if (Math.round(prediction[i1]) != output[i1]) return false;
            return true;
        };
        var start = Date.now();
        while(trial < iterations && (success < criterion || trial % 1000 != 0)){
            var sequence = [], sequenceLength = length - prompts.length;
            for(i = 0; i < sequenceLength; i++){
                var any = Math.random() * distractors.length | 0;
                sequence.push(distractors[any]);
            }
            var indexes = [], positions = [];
            for(i = 0; i < prompts.length; i++){
                indexes.push(Math.random() * targets.length | 0);
                positions.push(noRepeat(sequenceLength, positions));
            }
            positions = positions.sort();
            for(i = 0; i < prompts.length; i++){
                sequence[positions[i]] = targets[indexes[i]];
                sequence.push(prompts[i]);
            }
            var distractorsCorrect;
            var targetsCorrect = distractorsCorrect = 0;
            error = 0;
            for(i = 0; i < length; i++){
                var input = [];
                for(j = 0; j < symbols; j++)input[j] = 0;
                input[sequence[i]] = 1;
                var output = [];
                for(j = 0; j < targets.length; j++)output[j] = 0;
                if (i >= sequenceLength) {
                    var index = i - sequenceLength;
                    output[indexes[index]] = 1;
                }
                var prediction = this.network.activate(input);
                if (equal(prediction, output)) {
                    if (i < sequenceLength) distractorsCorrect++;
                    else targetsCorrect++;
                } else {
                    this.network.propagate(rate, output);
                }
                error += cost4(output, prediction);
                if (distractorsCorrect + targetsCorrect == length) correct++;
            }
            if (trial % 1000 == 0) correct = 0;
            trial++;
            var divideError = trial % 1000;
            divideError = divideError == 0 ? 1000 : divideError;
            success = correct / divideError;
            error /= length;
            if (log && trial % log == 0) console.log('iterations:', trial, ' success:', success, ' correct:', correct, ' time:', Date.now() - start, ' error:', error);
            if (schedule.do && schedule.every && trial % schedule.every == 0) schedule.do({
                iterations: trial,
                success: success,
                error: error,
                time: Date.now() - start,
                correct: correct
            });
        }
        return {
            iterations: trial,
            success: success,
            error: error,
            time: Date.now() - start
        };
    }
    ERG(options) {
        options = options || {
        };
        var iterations = options.iterations || 150000;
        var criterion = options.error || 0.05;
        var rate = options.rate || 0.1;
        var log = options.log || 500;
        var cost5 = options.cost || this.cost || Trainer.cost.CROSS_ENTROPY;
        var Node1 = function() {
            this.paths = [];
        };
        Node1.prototype = {
            connect: function(node, value) {
                this.paths.push({
                    node: node,
                    value: value
                });
                return this;
            },
            any: function() {
                if (this.paths.length == 0) return false;
                var index = Math.random() * this.paths.length | 0;
                return this.paths[index];
            },
            test: function(value) {
                for(var i in this.paths)if (this.paths[i].value == value) return this.paths[i];
                return false;
            }
        };
        var reberGrammar = function() {
            var output = new Node1();
            var n1 = new Node1().connect(output, 'E');
            var n2 = new Node1().connect(n1, 'S');
            var n3 = new Node1().connect(n1, 'V').connect(n2, 'P');
            var n4 = new Node1().connect(n2, 'X');
            n4.connect(n4, 'S');
            var n5 = new Node1().connect(n3, 'V');
            n5.connect(n5, 'T');
            n2.connect(n5, 'X');
            var n6 = new Node1().connect(n4, 'T').connect(n5, 'P');
            var input = new Node1().connect(n6, 'B');
            return {
                input: input,
                output: output
            };
        };
        var embededReberGrammar = function() {
            var reber1 = reberGrammar();
            var reber2 = reberGrammar();
            var output = new Node1();
            var n1 = (new Node1).connect(output, 'E');
            reber1.output.connect(n1, 'T');
            reber2.output.connect(n1, 'P');
            var n2 = (new Node1).connect(reber1.input, 'P').connect(reber2.input, 'T');
            var input = (new Node1).connect(n2, 'B');
            return {
                input: input,
                output: output
            };
        };
        var generate = function() {
            var node = embededReberGrammar().input;
            var next = node.any();
            var str = '';
            while(next){
                str += next.value;
                next = next.node.any();
            }
            return str;
        };
        var test = function(str) {
            var node = embededReberGrammar().input;
            var i = 0;
            var ch = str.charAt(i);
            while(i < str.length){
                var next = node.test(ch);
                if (!next) return false;
                node = next.node;
                ch = str.charAt(++i);
            }
            return true;
        };
        var different = function(array1, array2) {
            var max1 = 0;
            var i1 = -1;
            var max2 = 0;
            var i2 = -1;
            for(var i in array1){
                if (array1[i] > max1) {
                    max1 = array1[i];
                    i1 = i;
                }
                if (array2[i] > max2) {
                    max2 = array2[i];
                    i2 = i;
                }
            }
            return i1 != i2;
        };
        var iteration = 0;
        var error = 1;
        var table = {
            'B': 0,
            'P': 1,
            'T': 2,
            'X': 3,
            'S': 4,
            'E': 5
        };
        var start = Date.now();
        while(iteration < iterations && error > criterion){
            var i = 0;
            error = 0;
            var sequence = generate();
            var read = sequence.charAt(i);
            var predict = sequence.charAt(i + 1);
            while(i < sequence.length - 1){
                var input = [];
                var target = [];
                for(var j = 0; j < 6; j++){
                    input[j] = 0;
                    target[j] = 0;
                }
                input[table[read]] = 1;
                target[table[predict]] = 1;
                var output = this.network.activate(input);
                if (different(output, target)) this.network.propagate(rate, target);
                read = sequence.charAt(++i);
                predict = sequence.charAt(i + 1);
                error += cost5(target, output);
            }
            error /= sequence.length;
            iteration++;
            if (iteration % log == 0) {
                console.log('iterations:', iteration, ' time:', Date.now() - start, ' error:', error);
            }
        }
        return {
            iterations: iteration,
            error: error,
            time: Date.now() - start,
            test: test,
            generate: generate
        };
    }
    timingTask(options) {
        if (this.network.inputs() != 2 || this.network.outputs() != 1) throw new Error('Invalid Network: must have 2 inputs and one output');
        if (typeof options == 'undefined') options = {
        };
        function getSamples(trainingSize, testSize) {
            var size = trainingSize + testSize;
            var t = 0;
            var set = [];
            for(var i = 0; i < size; i++){
                set.push({
                    input: [
                        0,
                        0
                    ],
                    output: [
                        0
                    ]
                });
            }
            while(t < size - 20){
                var n = Math.round(Math.random() * 20);
                set[t].input[0] = 1;
                for(var j = t; j <= t + n; j++){
                    set[j].input[1] = n / 20;
                    set[j].output[0] = 0.5;
                }
                t += n;
                n = Math.round(Math.random() * 20);
                for(var k = t + 1; k <= t + n && k < size; k++)set[k].input[1] = set[t].input[1];
                t += n;
            }
            var trainingSet = [];
            var testSet = [];
            for(var l = 0; l < size; l++)(l < trainingSize ? trainingSet : testSet).push(set[l]);
            return {
                train: trainingSet,
                test: testSet
            };
        }
        var iterations = options.iterations || 200;
        var error = options.error || 0.005;
        var rate = options.rate || [
            0.03,
            0.02
        ];
        var log = options.log === false ? false : options.log || 10;
        var cost6 = options.cost || this.cost || Trainer.cost.MSE;
        var trainingSamples = options.trainSamples || 7000;
        var testSamples = options.trainSamples || 1000;
        var samples = getSamples(trainingSamples, testSamples);
        var result = this.train(samples.train, {
            rate: rate,
            log: log,
            iterations: iterations,
            error: error,
            cost: cost6
        });
        return {
            train: result,
            test: this.test(samples.test)
        };
    }
}
class Network {
    constructor(layers1){
        if (typeof layers1 != 'undefined') {
            this.layers = {
                input: layers1.input || null,
                hidden: layers1.hidden || [],
                output: layers1.output || null
            };
            this.optimized = null;
        }
    }
    activate(input) {
        if (this.optimized === false) {
            this.layers.input.activate(input);
            for(var i = 0; i < this.layers.hidden.length; i++)this.layers.hidden[i].activate();
            return this.layers.output.activate();
        } else {
            if (this.optimized == null) this.optimize();
            return this.optimized.activate(input);
        }
    }
    propagate(rate, target) {
        if (this.optimized === false) {
            this.layers.output.propagate(rate, target);
            for(let i = this.layers.hidden.length - 1; i >= 0; i--)this.layers.hidden[i].propagate(rate);
        } else {
            if (this.optimized == null) this.optimize();
            this.optimized.propagate(rate, target);
        }
    }
    project(unit, type, weights) {
        if (this.optimized) this.optimized.reset();
        if (unit instanceof Network) return this.layers.output.project(unit.layers.input, type, weights);
        if (unit instanceof Layer) return this.layers.output.project(unit, type, weights);
        throw new Error('Invalid argument, you can only project connections to LAYERS and NETWORKS!');
    }
    gate(connection, type) {
        if (this.optimized) this.optimized.reset();
        this.layers.output.gate(connection, type);
    }
    clear() {
        this.restore();
        let inputLayer = this.layers.input, outputLayer = this.layers.output;
        inputLayer.clear();
        for(let i = 0; i < this.layers.hidden.length; i++){
            this.layers.hidden[i].clear();
        }
        outputLayer.clear();
        if (this.optimized) this.optimized.reset();
    }
    reset() {
        this.restore();
        let inputLayer = this.layers.input, outputLayer = this.layers.output;
        inputLayer.reset();
        for(let i = 0; i < this.layers.hidden.length; i++){
            this.layers.hidden[i].reset();
        }
        outputLayer.reset();
        if (this.optimized) this.optimized.reset();
    }
    optimize() {
        let that = this;
        let optimized = {
        };
        let neurons2 = this.neurons();
        for(let i = 0; i < neurons2.length; i++){
            let neuron = neurons2[i].neuron;
            let layer = neurons2[i].layer;
            while(neuron.neuron)neuron = neuron.neuron;
            optimized = neuron.optimize(optimized, layer);
        }
        for(let i1 = 0; i1 < optimized.propagation_sentences.length; i1++)optimized.propagation_sentences[i1].reverse();
        optimized.propagation_sentences.reverse();
        let hardcode = '';
        hardcode += 'var F = Float64Array ? new Float64Array(' + optimized.memory + ') : []; ';
        for(let i2 in optimized.variables)hardcode += 'F[' + optimized.variables[i2].id + '] = ' + (optimized.variables[i2].value || 0) + '; ';
        hardcode += 'var activate = function(input){\n';
        for(let i3 = 0; i3 < optimized.inputs.length; i3++)hardcode += 'F[' + optimized.inputs[i3] + '] = input[' + i3 + ']; ';
        for(let i4 = 0; i4 < optimized.activation_sentences.length; i4++){
            if (optimized.activation_sentences[i4].length > 0) {
                for(let j = 0; j < optimized.activation_sentences[i4].length; j++){
                    hardcode += optimized.activation_sentences[i4][j].join(' ');
                    hardcode += optimized.trace_sentences[i4][j].join(' ');
                }
            }
        }
        hardcode += ' var output = []; ';
        for(let i5 = 0; i5 < optimized.outputs.length; i5++)hardcode += 'output[' + i5 + '] = F[' + optimized.outputs[i5] + ']; ';
        hardcode += 'return output; }; ';
        hardcode += 'var propagate = function(rate, target){\n';
        hardcode += 'F[' + optimized.variables.rate.id + '] = rate; ';
        for(let i6 = 0; i6 < optimized.targets.length; i6++)hardcode += 'F[' + optimized.targets[i6] + '] = target[' + i6 + ']; ';
        for(let i7 = 0; i7 < optimized.propagation_sentences.length; i7++)for(let j = 0; j < optimized.propagation_sentences[i7].length; j++)hardcode += optimized.propagation_sentences[i7][j].join(' ') + ' ';
        hardcode += ' };\n';
        hardcode += 'var ownership = function(memoryBuffer){\nF = memoryBuffer;\nthis.memory = F;\n};\n';
        hardcode += 'return {\nmemory: F,\nactivate: activate,\npropagate: propagate,\nownership: ownership\n};';
        hardcode = hardcode.split(';').join(';\n');
        let constructor = new Function(hardcode);
        let network1 = constructor();
        network1.data = {
            variables: optimized.variables,
            activate: optimized.activation_sentences,
            propagate: optimized.propagation_sentences,
            trace: optimized.trace_sentences,
            inputs: optimized.inputs,
            outputs: optimized.outputs,
            check_activation: this.activate,
            check_propagation: this.propagate
        };
        network1.reset = ()=>{
            if (that.optimized) {
                that.optimized = null;
                that.activate = network1.data.check_activation;
                that.propagate = network1.data.check_propagation;
            }
        };
        this.optimized = network1;
        this.activate = network1.activate;
        this.propagate = network1.propagate;
    }
    restore() {
        if (!this.optimized) return;
        let optimized = this.optimized;
        let getValue = ()=>{
            let args = Array.prototype.slice.call(arguments);
            let unit = args.shift();
            let prop = args.pop();
            let id = prop + '_';
            for(let property in args)id += args[property] + '_';
            id += unit.ID;
            let memory = optimized.memory;
            let variables = optimized.data.variables;
            if (id in variables) return memory[variables[id].id];
            return 0;
        };
        let list = this.neurons();
        for(let i = 0; i < list.length; i++){
            let neuron = list[i].neuron;
            while(neuron.neuron)neuron = neuron.neuron;
            neuron.state = getValue(neuron, 'state');
            neuron.old = getValue(neuron, 'old');
            neuron.activation = getValue(neuron, 'activation');
            neuron.bias = getValue(neuron, 'bias');
            for(let input in neuron.trace.elegibility)neuron.trace.elegibility[input] = getValue(neuron, 'trace', 'elegibility', input);
            for(let gated in neuron.trace.extended)for(let input1 in neuron.trace.extended[gated])neuron.trace.extended[gated][input1] = getValue(neuron, 'trace', 'extended', gated, input1);
            for(let j in neuron.connections.projected){
                let connection = neuron.connections.projected[j];
                connection.weight = getValue(connection, 'weight');
                connection.gain = getValue(connection, 'gain');
            }
        }
    }
    neurons() {
        let neurons2 = [];
        let inputLayer = this.layers.input.neurons(), outputLayer = this.layers.output.neurons();
        for(let i = 0; i < inputLayer.length; i++){
            neurons2.push({
                neuron: inputLayer[i],
                layer: 'input'
            });
        }
        for(let i1 = 0; i1 < this.layers.hidden.length; i1++){
            let hiddenLayer = this.layers.hidden[i1].neurons();
            for(let j = 0; j < hiddenLayer.length; j++)neurons2.push({
                neuron: hiddenLayer[j],
                layer: i1
            });
        }
        for(let i2 = 0; i2 < outputLayer.length; i2++){
            neurons2.push({
                neuron: outputLayer[i2],
                layer: 'output'
            });
        }
        return neurons2;
    }
    inputs() {
        return this.layers.input.size;
    }
    outputs() {
        return this.layers.output.size;
    }
    set(layers) {
        this.layers = {
            input: layers.input || null,
            hidden: layers.hidden || [],
            output: layers.output || null
        };
        if (this.optimized) this.optimized.reset();
    }
    setOptimize(bool) {
        this.restore();
        if (this.optimized) this.optimized.reset();
        this.optimized = bool ? null : false;
    }
    toJSON(ignoreTraces) {
        this.restore();
        let list = this.neurons();
        let neurons2 = [];
        let connections1 = [];
        let ids = {
        };
        for(let i = 0; i < list.length; i++){
            let neuron = list[i].neuron;
            while(neuron.neuron)neuron = neuron.neuron;
            ids[neuron.ID] = i;
            let copy = {
                trace: {
                    elegibility: {
                    },
                    extended: {
                    }
                },
                state: neuron.state,
                old: neuron.old,
                activation: neuron.activation,
                bias: neuron.bias,
                layer: list[i].layer
            };
            copy.squash = neuron.squash == Neuron.squash.LOGISTIC ? 'LOGISTIC' : neuron.squash == Neuron.squash.TANH ? 'TANH' : neuron.squash == Neuron.squash.IDENTITY ? 'IDENTITY' : neuron.squash == Neuron.squash.HLIM ? 'HLIM' : neuron.squash == Neuron.squash.RELU ? 'RELU' : null;
            neurons2.push(copy);
        }
        for(let i1 = 0; i1 < list.length; i1++){
            let neuron = list[i1].neuron;
            while(neuron.neuron)neuron = neuron.neuron;
            for(let j in neuron.connections.projected){
                let connection = neuron.connections.projected[j];
                connections1.push({
                    from: ids[connection.from.ID],
                    to: ids[connection.to.ID],
                    weight: connection.weight,
                    gater: connection.gater ? ids[connection.gater.ID] : null
                });
            }
            if (neuron.selfconnected()) {
                connections1.push({
                    from: ids[neuron.ID],
                    to: ids[neuron.ID],
                    weight: neuron.selfconnection.weight,
                    gater: neuron.selfconnection.gater ? ids[neuron.selfconnection.gater.ID] : null
                });
            }
        }
        return {
            neurons: neurons2,
            connections: connections1
        };
    }
    toDot(edgeConnection) {
        if (!typeof edgeConnection) edgeConnection = false;
        let code = 'digraph nn {\n    rankdir = BT\n';
        let layers2 = [
            this.layers.input
        ].concat(this.layers.hidden, this.layers.output);
        for(let i = 0; i < layers2.length; i++){
            for(let j = 0; j < layers2[i].connectedTo.length; j++){
                let connection = layers2[i].connectedTo[j];
                let layerTo = connection.to;
                let size = connection.size;
                let layerID = layers2.indexOf(layers2[i]);
                let layerToID = layers2.indexOf(layerTo);
                if (edgeConnection) {
                    if (connection.gatedfrom.length) {
                        let fakeNode = 'fake' + layerID + '_' + layerToID;
                        code += '    ' + fakeNode + ' [label = "", shape = point, width = 0.01, height = 0.01]\n';
                        code += '    ' + layerID + ' -> ' + fakeNode + ' [label = ' + size + ', arrowhead = none]\n';
                        code += '    ' + fakeNode + ' -> ' + layerToID + '\n';
                    } else code += '    ' + layerID + ' -> ' + layerToID + ' [label = ' + size + ']\n';
                    for(let from1 in connection.gatedfrom){
                        let layerfrom = connection.gatedfrom[from1].layer;
                        let layerfromID = layers2.indexOf(layerfrom);
                        code += '    ' + layerfromID + ' -> ' + fakeNode + ' [color = blue]\n';
                    }
                } else {
                    code += '    ' + layerID + ' -> ' + layerToID + ' [label = ' + size + ']\n';
                    for(let from1 in connection.gatedfrom){
                        let layerfrom = connection.gatedfrom[from1].layer;
                        let layerfromID = layers2.indexOf(layerfrom);
                        code += '    ' + layerfromID + ' -> ' + layerToID + ' [color = blue]\n';
                    }
                }
            }
        }
        code += '}\n';
        return {
            code: code,
            link: 'https://chart.googleapis.com/chart?chl=' + escape(code.replace('/ /g', '+')) + '&cht=gv'
        };
    }
    standalone() {
        if (!this.optimized) this.optimize();
        let data = this.optimized.data;
        let activation = 'function (input) {\n';
        for(let i = 0; i < data.inputs.length; i++)activation += 'F[' + data.inputs[i] + '] = input[' + i + '];\n';
        for(let i1 = 0; i1 < data.activate.length; i1++){
            for(let j = 0; j < data.activate[i1].length; j++)activation += data.activate[i1][j].join('') + '\n';
        }
        activation += 'var output = [];\n';
        for(let i2 = 0; i2 < data.outputs.length; i2++)activation += 'output[' + i2 + '] = F[' + data.outputs[i2] + '];\n';
        activation += 'return output;\n}';
        let memory = activation.match(/F\[(\d+)\]/g);
        let dimension = 0;
        let ids = {
        };
        for(let i3 = 0; i3 < memory.length; i3++){
            let tmp = memory[i3].match(/\d+/)[0];
            if (!(tmp in ids)) {
                ids[tmp] = dimension++;
            }
        }
        let hardcode = 'F = {\n';
        for(let i4 in ids)hardcode += ids[i4] + ': ' + this.optimized.memory[i4] + ',\n';
        hardcode = hardcode.substring(0, hardcode.length - 2) + '\n};\n';
        hardcode = 'var run = ' + activation.replace(/F\[(\d+)]/g, function(index) {
            return 'F[' + ids[index.match(/\d+/)[0]] + ']';
        }).replace('{\n', '{\n' + hardcode + '') + ';\n';
        hardcode += 'return run';
        return new Function(hardcode)();
    }
    worker(memory, set, options) {
        let workerOptions = {
        };
        if (options) workerOptions = options;
        workerOptions.rate = workerOptions.rate || 0.2;
        workerOptions.iterations = workerOptions.iterations || 100000;
        workerOptions.error = workerOptions.error || 0.005;
        workerOptions.cost = workerOptions.cost || null;
        workerOptions.crossValidate = workerOptions.crossValidate || null;
        let costFunction = '// REPLACED BY WORKER\nvar cost = ' + (options && options.cost || this.cost || Trainer.cost.MSE) + ';\n';
        let workerFunction = Network.getWorkerSharedFunctions();
        workerFunction = workerFunction.replace(/var cost = options && options\.cost \|\| this\.cost \|\| Trainer\.cost\.MSE;/g, costFunction);
        workerFunction = workerFunction.replace('return results;', 'postMessage({action: "done", message: results, memoryBuffer: F}, [F.buffer]);');
        workerFunction = workerFunction.replace('console.log(\'iterations\', iterations, \'error\', error, \'rate\', currentRate)', 'postMessage({action: \'log\', message: {\n' + 'iterations: iterations,\n' + 'error: error,\n' + 'rate: currentRate\n' + '}\n' + '})');
        workerFunction = workerFunction.replace('abort = this.schedule.do({ error: error, iterations: iterations, rate: currentRate })', 'postMessage({action: \'schedule\', message: {\n' + 'iterations: iterations,\n' + 'error: error,\n' + 'rate: currentRate\n' + '}\n' + '})');
        if (!this.optimized) this.optimize();
        let hardcode = 'var inputs = ' + this.optimized.data.inputs.length + ';\n';
        hardcode += 'var outputs = ' + this.optimized.data.outputs.length + ';\n';
        hardcode += 'var F =  new Float64Array([' + this.optimized.memory.toString() + ']);\n';
        hardcode += 'var activate = ' + this.optimized.activate.toString() + ';\n';
        hardcode += 'var propagate = ' + this.optimized.propagate.toString() + ';\n';
        hardcode += 'onmessage = function(e) {\n' + 'if (e.data.action == \'startTraining\') {\n' + 'train(' + JSON.stringify(set) + ',' + JSON.stringify(workerOptions) + ');\n' + '}\n' + '}';
        let workerSourceCode = workerFunction + '\n' + hardcode;
        let blob = new Blob([
            workerSourceCode
        ]);
        let blobURL = window.URL.createObjectURL(blob);
        return new Worker(blobURL);
    }
    clone() {
        return Network.fromJSON(this.toJSON());
    }
    static getWorkerSharedFunctions() {
        if (typeof Network._SHARED_WORKER_FUNCTIONS !== 'undefined') return Network._SHARED_WORKER_FUNCTIONS;
        let train_f = Trainer.prototype.train.toString();
        train_f = train_f.replace(/this._trainSet/g, '_trainSet');
        train_f = train_f.replace(/this.test/g, 'test');
        train_f = train_f.replace(/this.crossValidate/g, 'crossValidate');
        train_f = train_f.replace('crossValidate = true', '// REMOVED BY WORKER');
        let _trainSet_f = Trainer.prototype._trainSet.toString().replace(/this.network./g, '');
        let test_f = Trainer.prototype.test.toString().replace(/this.network./g, '');
        return Network._SHARED_WORKER_FUNCTIONS = train_f + '\n' + _trainSet_f + '\n' + test_f;
    }
    static fromJSON(json) {
        let neurons2 = [];
        let layers2 = {
            input: new Layer(),
            hidden: [],
            output: new Layer()
        };
        for(let i = 0; i < json.neurons.length; i++){
            let config = json.neurons[i];
            let neuron = new Neuron();
            neuron.trace.elegibility = {
            };
            neuron.trace.extended = {
            };
            neuron.state = config.state;
            neuron.old = config.old;
            neuron.activation = config.activation;
            neuron.bias = config.bias;
            neuron.squash = config.squash in Neuron.squash ? Neuron.squash[config.squash] : Neuron.squash.LOGISTIC;
            neurons2.push(neuron);
            if (config.layer == 'input') layers2.input.add(neuron);
            else if (config.layer == 'output') layers2.output.add(neuron);
            else {
                if (typeof layers2.hidden[config.layer] == 'undefined') layers2.hidden[config.layer] = new Layer();
                layers2.hidden[config.layer].add(neuron);
            }
        }
        for(let i1 = 0; i1 < json.connections.length; i1++){
            let config = json.connections[i1];
            let from1 = neurons2[config.from];
            let to1 = neurons2[config.to];
            let weight2 = config.weight;
            let gater = neurons2[config.gater];
            let connection = from1.project(to1, weight2);
            if (gater) gater.gate(connection);
        }
        return new Network(layers2);
    }
}
const connectionType1 = {
    ALL_TO_ALL: "ALL TO ALL",
    ONE_TO_ONE: "ONE TO ONE",
    ALL_TO_ELSE: "ALL TO ELSE"
};
let connections1 = 0;
const gateType1 = {
    INPUT: "INPUT",
    OUTPUT: "OUTPUT",
    ONE_TO_ONE: "ONE TO ONE"
};
class Layer {
    static connectionType = connectionType1;
    static gateType = gateType1;
    constructor(size1){
        this.size = size1 | 0;
        this.list = [];
        this.connectedTo = [];
        while(size1--){
            let neuron = new Neuron();
            this.list.push(neuron);
        }
    }
    activate(input) {
        let activations = [];
        if (typeof input != "undefined") {
            if (input.length != this.size) {
                throw new Error("INPUT size and LAYER size must be the same to activate!");
            }
            for(let id in this.list){
                let neuron = this.list[id];
                let activation = neuron.activate(input[id]);
                activations.push(activation);
            }
        } else {
            for(let id in this.list){
                let neuron = this.list[id];
                let activation = neuron.activate();
                activations.push(activation);
            }
        }
        return activations;
    }
    propagate(rate, target) {
        if (typeof target != "undefined") {
            if (target.length != this.size) {
                throw new Error("TARGET size and LAYER size must be the same to propagate!");
            }
            for(let id = this.list.length - 1; id >= 0; id--){
                let neuron = this.list[id];
                neuron.propagate(rate, target[id]);
            }
        } else {
            for(let id = this.list.length - 1; id >= 0; id--){
                let neuron = this.list[id];
                neuron.propagate(rate);
            }
        }
    }
    project(layer, type, weights) {
        if (layer instanceof Network) {
            layer = layer.layers.input;
        }
        if (layer instanceof Layer) {
            if (!this.connected(layer)) {
                return new LayerConnection(this, layer, type, weights);
            }
        } else {
            throw new Error("Invalid argument, you can only project connections to LAYERS and NETWORKS!");
        }
    }
    gate(connection, type) {
        if (type == Layer.gateType.INPUT) {
            if (connection.to.size != this.size) {
                throw new Error("GATER layer and CONNECTION.TO layer must be the same size in order to gate!");
            }
            for(let id in connection.to.list){
                let neuron = connection.to.list[id];
                let gater = this.list[id];
                for(let input in neuron.connections.inputs){
                    let gated = neuron.connections.inputs[input];
                    if (gated.ID in connection.connections) {
                        gater.gate(gated);
                    }
                }
            }
        } else if (type == Layer.gateType.OUTPUT) {
            if (connection.from.size != this.size) {
                throw new Error("GATER layer and CONNECTION.FROM layer must be the same size in order to gate!");
            }
            for(var id in connection.from.list){
                let neuron = connection.from.list[id];
                let gater = this.list[id];
                for(let projected in neuron.connections.projected){
                    let gated = neuron.connections.projected[projected];
                    if (gated.ID in connection.connections) {
                        gater.gate(gated);
                    }
                }
            }
        } else if (type == Layer.gateType.ONE_TO_ONE) {
            if (connection.size != this.size) {
                throw new Error("The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!");
            }
            for(let id in connection.list){
                let gater = this.list[id];
                let gated = connection.list[id];
                gater.gate(gated);
            }
        }
        connection.gatedfrom.push({
            layer: this,
            type: type
        });
    }
    selfconnected() {
        for(let id in this.list){
            let neuron = this.list[id];
            if (!neuron.selfconnected()) {
                return false;
            }
        }
        return true;
    }
    connected(layer) {
        let connections2 = 0;
        for(let here in this.list){
            for(let there in layer.list){
                let from1 = this.list[here];
                let to1 = layer.list[there];
                let connected = from1.connected(to1);
                if (connected.type == "projected") {
                    connections2++;
                }
            }
        }
        if (connections2 == this.size * layer.size) {
            return Layer.connectionType.ALL_TO_ALL;
        }
        connections2 = 0;
        for(let neuron in this.list){
            let from1 = this.list[neuron];
            let to1 = layer.list[neuron];
            let connected = from1.connected(to1);
            if (connected.type == "projected") {
                connections2++;
            }
        }
        if (connections2 == this.size) {
            return Layer.connectionType.ONE_TO_ONE;
        }
    }
    clear() {
        for(let id in this.list){
            let neuron = this.list[id];
            neuron.clear();
        }
    }
    reset() {
        for(let id in this.list){
            let neuron = this.list[id];
            neuron.reset();
        }
    }
    neurons() {
        return this.list;
    }
    add(neuron) {
        neuron = neuron || new Neuron();
        this.list.push(neuron);
        this.size++;
    }
    set(options) {
        options = options || {
        };
        for(let i in this.list){
            let neuron = this.list[i];
            if (options.label) {
                neuron.label = options.label + "_" + neuron.ID;
            }
            if (options.squash) {
                neuron.squash = options.squash;
            }
            if (options.bias) {
                neuron.bias = options.bias;
            }
        }
        return this;
    }
}
class LayerConnection {
    type;
    ID;
    from;
    to;
    selfconnection;
    connections;
    list;
    size;
    gatedfrom;
    constructor(fromLayer, toLayer, type, weights){
        this.type = type;
        this.ID = LayerConnection.uid();
        this.from = fromLayer;
        this.to = toLayer;
        this.selfconnection = toLayer == fromLayer;
        this.connections = {
        };
        this.list = [];
        this.size = 0;
        this.gatedfrom = [];
        if (typeof this.type == "undefined") {
            if (fromLayer == toLayer) {
                this.type = Layer.connectionType.ONE_TO_ONE;
            } else {
                this.type = Layer.connectionType.ALL_TO_ALL;
            }
        }
        if (this.type == Layer.connectionType.ALL_TO_ALL || this.type == Layer.connectionType.ALL_TO_ELSE) {
            for(let here in this.from.list){
                for(let there in this.to.list){
                    let from2 = this.from.list[here];
                    let to2 = this.to.list[there];
                    if (this.type == Layer.connectionType.ALL_TO_ELSE && from2 == to2) {
                        continue;
                    }
                    let connection = from2.project(to2, weights);
                    this.connections[connection.ID] = connection;
                    this.size = this.list.push(connection);
                }
            }
        } else if (this.type == Layer.connectionType.ONE_TO_ONE) {
            for(let neuron in this.from.list){
                let from2 = this.from.list[neuron];
                let to2 = this.to.list[neuron];
                let connection = from2.project(to2, weights);
                this.connections[connection.ID] = connection;
                this.size = this.list.push(connection);
            }
        }
        fromLayer.connectedTo.push(this);
    }
    static uid() {
        return connections1++;
    }
}
class Perceptron extends Network {
    constructor(){
        super();
        let args = Array.prototype.slice.call(arguments);
        if (args.length < 3) throw new Error('not enough layers (minimum 3) !!');
        let inputs = args.shift();
        let outputs1 = args.pop();
        let layers2 = args;
        let input = new Layer(inputs);
        let hidden = [];
        let output = new Layer(outputs1);
        let previous = input;
        for(let i = 0; i < layers2.length; i++){
            let size2 = layers2[i];
            let layer = new Layer(size2);
            hidden.push(layer);
            previous.project(layer);
            previous = layer;
        }
        previous.project(output);
        this.set({
            input: input,
            hidden: hidden,
            output: output
        });
    }
}
class LSTM extends Network {
    constructor(){
        super();
        let args1 = Array.prototype.slice.call(arguments);
        if (args1.length < 3) throw new Error("not enough layers (minimum 3) !!");
        let last = args1.pop();
        let option = {
            peepholes: Layer.connectionType.ALL_TO_ALL,
            hiddenToHidden: false,
            outputToHidden: false,
            outputToGates: false,
            inputToOutput: true
        };
        if (typeof last != 'number') {
            let outputs1 = args1.pop();
            if (last.hasOwnProperty('peepholes')) option.peepholes = last.peepholes;
            if (last.hasOwnProperty('hiddenToHidden')) option.hiddenToHidden = last.hiddenToHidden;
            if (last.hasOwnProperty('outputToHidden')) option.outputToHidden = last.outputToHidden;
            if (last.hasOwnProperty('outputToGates')) option.outputToGates = last.outputToGates;
            if (last.hasOwnProperty('inputToOutput')) option.inputToOutput = last.inputToOutput;
        } else {
            let outputs1 = last;
        }
        let inputs1 = args1.shift();
        let layers3 = args1;
        let inputLayer = new Layer(inputs1);
        let hiddenLayers = [];
        let outputLayer = new Layer(outputs);
        let previous1 = null;
        for(let i1 = 0; i1 < layers3.length; i1++){
            let size2 = layers3[i1];
            let inputGate = new Layer(size2).set({
                bias: 1
            });
            let forgetGate = new Layer(size2).set({
                bias: 1
            });
            let memoryCell = new Layer(size2);
            let outputGate = new Layer(size2).set({
                bias: 1
            });
            hiddenLayers.push(inputGate);
            hiddenLayers.push(forgetGate);
            hiddenLayers.push(memoryCell);
            hiddenLayers.push(outputGate);
            let input1 = inputLayer.project(memoryCell);
            inputLayer.project(inputGate);
            inputLayer.project(forgetGate);
            inputLayer.project(outputGate);
            if (previous1 != null) {
                let cell = previous1.project(memoryCell);
                previous1.project(inputGate);
                previous1.project(forgetGate);
                previous1.project(outputGate);
            }
            let output1 = memoryCell.project(outputLayer);
            let self = memoryCell.project(memoryCell);
            if (option.hiddenToHidden) memoryCell.project(memoryCell, Layer.connectionType.ALL_TO_ELSE);
            if (option.outputToHidden) outputLayer.project(memoryCell);
            if (option.outputToGates) {
                outputLayer.project(inputGate);
                outputLayer.project(outputGate);
                outputLayer.project(forgetGate);
            }
            memoryCell.project(inputGate, option.peepholes);
            memoryCell.project(forgetGate, option.peepholes);
            memoryCell.project(outputGate, option.peepholes);
            inputGate.gate(input1, Layer.gateType.INPUT);
            forgetGate.gate(self, Layer.gateType.ONE_TO_ONE);
            outputGate.gate(output1, Layer.gateType.OUTPUT);
            if (previous1 != null) inputGate.gate(cell, Layer.gateType.INPUT);
            previous1 = memoryCell;
        }
        if (option.inputToOutput) inputLayer.project(outputLayer);
        this.set({
            input: inputLayer,
            hidden: hiddenLayers,
            output: outputLayer
        });
    }
}
class Liquid extends Network {
    constructor(inputs2, hidden1, outputs2, connections3, gates){
        super();
        let inputLayer1 = new Layer(inputs2);
        let hiddenLayer = new Layer(hidden1);
        let outputLayer1 = new Layer(outputs2);
        let neurons2 = hiddenLayer.neurons();
        let connectionList = [];
        for(let i2 = 0; i2 < connections3; i2++){
            let from2 = Math.random() * neurons2.length | 0;
            let to2 = Math.random() * neurons2.length | 0;
            let connection = neurons2[from2].project(neurons2[to2]);
            connectionList.push(connection);
        }
        for(let j = 0; j < gates; j++){
            let gater = Math.random() * neurons2.length | 0;
            let connection = Math.random() * connectionList.length | 0;
            neurons2[gater].gate(connectionList[connection]);
        }
        inputLayer1.project(hiddenLayer);
        hiddenLayer.project(outputLayer1);
        this.set({
            input: inputLayer1,
            hidden: [
                hiddenLayer
            ],
            output: outputLayer1
        });
    }
}
class Hopfield extends Network {
    constructor(size2){
        super();
        let inputLayer2 = new Layer(size2);
        let outputLayer2 = new Layer(size2);
        inputLayer2.project(outputLayer2, Layer.connectionType.ALL_TO_ALL);
        this.set({
            input: inputLayer2,
            hidden: [],
            output: outputLayer2
        });
        this.trainer = new Trainer(this);
    }
    learn(patterns) {
        let set = [];
        for(let p in patterns)set.push({
            input: patterns[p],
            output: patterns[p]
        });
        return this.trainer.train(set, {
            iterations: 500000,
            error: 0.00005,
            rate: 1
        });
    }
    feed(pattern) {
        let output1 = this.activate(pattern);
        pattern = [];
        for(let i3 in output1)pattern[i3] = output1[i3] > 0.5 ? 1 : 0;
        return pattern;
    }
}
const mod = function() {
    return {
        Perceptron: Perceptron,
        LSTM: LSTM,
        Liquid: Liquid,
        Hopfield: Hopfield
    };
}();
class Vector {
    x;
    y;
    constructor(x1, y1){
        this.x = x1;
        this.y = y1;
    }
    set(x, y) {
        this.x = x;
        this.y = y;
        return this;
    }
    add(v) {
        this.x += v.x;
        this.y += v.y;
        return this;
    }
    sub(v) {
        this.x -= v.x;
        this.y -= v.y;
        return this;
    }
    mul(s) {
        this.x *= s;
        this.y *= s;
        return this;
    }
    div(s) {
        this.x /= s;
        this.y /= s;
        return this;
    }
    get mag() {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }
    normalize() {
        this.mag && this.div(this.mag);
        return this;
    }
    get angle() {
        return Math.atan2(this.y, this.x);
    }
    setMag(m) {
        this.x = m * Math.cos(this.angle);
        this.y = m * Math.sin(this.angle);
        return this;
    }
    setAngle(a) {
        this.x = this.mag * Math.cos(a);
        this.y = this.mag * Math.sin(a);
        return this;
    }
    rotate(a) {
        this.setAngle(this.angle + a);
        return this;
    }
    limit(l) {
        if (this.mag > l) {
            this.setMag(l);
        }
        return this;
    }
    angleBetween(v) {
        return this.angle - v.angle;
    }
    dot(v) {
        return this.x * v.x + this.y * v.y;
    }
    lerp(v, amt) {
        this.x += (v.x - this.x) * amt;
        this.y += (v.y - this.y) * amt;
        return this;
    }
    dist(v) {
        const dx = this.x - v.x;
        const dy = this.y - v.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    get copy() {
        return new Vector(this.x, this.y);
    }
    random() {
        this.set(1, 1);
        this.setAngle(Math.random() * Math.PI * 2);
        return this;
    }
}
class Creature {
    world;
    network;
    mass = 0.3;
    maxSpeed = 2;
    maxForce = 0.2;
    HALF_PI = Math.PI * 0.5;
    TWO_PI = Math.PI * 2;
    lookRange;
    length;
    base;
    location;
    velocity = new Vector(0, 0);
    acceleration = new Vector(0, 0);
    color = 16777215;
    constructor(world, x2, y2){
        this.world = world;
        this.network = new mod.Perceptron(40, 25, 3);
        this.lookRange = this.mass * 200;
        this.length = this.mass * 10;
        this.base = this.length * 0.5;
        this.location = new Vector(x2, y2);
    }
    moveTo(networkOutput) {
        const force = new Vector(0, 0);
        const target = new Vector(networkOutput[0] * this.world.width, networkOutput[1] * this.world.height);
        const angle = networkOutput[2] * this.TWO_PI - Math.PI;
        const separation = this.separate(this.world.creatures);
        const alignment = this.align(this.world.creatures).setAngle(angle);
        const cohesion = this.seek(target);
        force.add(separation);
        force.add(alignment);
        force.add(cohesion);
        this.applyForce(force);
    }
    draw() {
        this.update();
        const ctx = this.world.context;
        ctx.lineWidth = 1;
        var angle = this.velocity.angle;
        const x11 = this.location.x + Math.cos(angle) * this.base;
        const y11 = this.location.y + Math.sin(angle) * this.base;
        const x21 = this.location.x + Math.cos(angle + this.HALF_PI) * this.base;
        const y21 = this.location.y + Math.sin(angle + this.HALF_PI) * this.base;
        const x3 = this.location.x + Math.cos(angle - this.HALF_PI) * this.base;
        const y3 = this.location.y + Math.sin(angle - this.HALF_PI) * this.base;
        ctx.lineWidth = 2;
        ctx.fillStyle = this.color;
        ctx.strokeStyle = this.color;
        ctx.beginPath();
        ctx.moveTo(x11, y11);
        ctx.lineTo(x21, y21);
        ctx.lineTo(x3, y3);
        ctx.stroke();
        ctx.fill();
        ctx.closePath();
    }
    update() {
        this.boundaries();
        this.velocity.add(this.acceleration);
        this.velocity.limit(this.maxSpeed);
        if (this.velocity.mag < 1.5) {
            this.velocity.setMag(1.5);
        }
        this.location.add(this.velocity);
        this.acceleration.mul(0);
    }
    applyForce(force) {
        this.acceleration.add(force);
    }
    boundaries() {
        if (this.location.x < 15) {
            this.applyForce(new Vector(this.maxForce * 2, 0));
        }
        if (this.location.x > this.world.width - 15) {
            this.applyForce(new Vector(-this.maxForce * 2, 0));
        }
        if (this.location.y < 15) {
            this.applyForce(new Vector(0, this.maxForce * 2));
        }
        if (this.location.y > this.world.height - 15) {
            this.applyForce(new Vector(0, -this.maxForce * 2));
        }
    }
    seek(target) {
        const seek = target.copy.sub(this.location);
        seek.normalize();
        seek.mul(this.maxSpeed);
        seek.sub(this.velocity).limit(0.3);
        return seek;
    }
    separate(neighboors) {
        const sum = new Vector(0, 0);
        let count = 0;
        for(const i3 in neighboors){
            if (neighboors[i3] != this) {
                const d = this.location.dist(neighboors[i3].location);
                if (d < 24 && d > 0) {
                    const diff = this.location.copy.sub(neighboors[i3].location);
                    diff.normalize();
                    diff.div(d);
                    sum.add(diff);
                    count++;
                }
            }
        }
        if (!count) return sum;
        sum.div(count);
        sum.normalize();
        sum.mul(this.maxSpeed);
        sum.sub(this.velocity);
        sum.limit(this.maxForce);
        return sum.mul(2);
    }
    align(neighboors) {
        const sum = new Vector(0, 0);
        let count = 0;
        for(const i3 in neighboors){
            if (neighboors[i3] != this) {
                sum.add(neighboors[i3].velocity);
                count++;
            }
        }
        sum.div(count);
        sum.normalize();
        sum.mul(this.maxSpeed);
        sum.sub(this.velocity).limit(this.maxSpeed);
        return sum.limit(0.1);
    }
    cohesion(neighboors) {
        const sum = new Vector(0, 0);
        let count = 0;
        for(const i3 in neighboors){
            if (neighboors[i3] != this) {
                sum.add(neighboors[i3].location);
                count++;
            }
        }
        sum.div(count);
        return sum;
    }
}
const init = ()=>{
    const canvas = document.getElementById("c");
    const ctx = canvas.getContext("2d");
    const num = 10;
    const fps = 100;
    const world1 = {
        width: document.getElementById("c").width,
        height: document.getElementById("c").height,
        creatures: [],
        context: ctx
    };
    for(let i3 = 0; i3 < 10; i3++){
        const x3 = Math.random() * world1.width;
        const y3 = Math.random() * world1.height;
        world1.creatures[i3] = new Creature(world1, x3, y3);
        world1.creatures[i3].velocity.random();
    }
    const targetX = (creature)=>{
        const cohesion = creature.cohesion(world1.creatures);
        return cohesion.x / world1.width;
    };
    const targetY = (creature)=>{
        const cohesion = creature.cohesion(world1.creatures);
        return cohesion.y / world1.height;
    };
    const targetAngle = (creature)=>{
        const alignment = creature.align(world1.creatures);
        return (alignment.angle + Math.PI) / (Math.PI * 2);
    };
    const loop = ()=>{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle, ctx.strokeStyle = 'white';
        ctx.stroke();
        const creatures = world1.creatures;
        creatures.forEach((creature)=>{
            const input1 = [];
            for(const i4 in creatures){
                input1.push(creatures[i4].location.x);
                input1.push(creatures[i4].location.y);
                input1.push(creatures[i4].velocity.x);
                input1.push(creatures[i4].velocity.y);
            }
            const output1 = creature.network.activate(input1);
            creature.moveTo(output1);
            const learningRate = 0.3;
            const target = [
                targetX(creature),
                targetY(creature),
                targetAngle(creature), 
            ];
            creature.network.propagate(0.3, target);
            creature.draw();
        });
        setTimeout(loop, 1000 / 100);
    };
    loop();
};
init();
