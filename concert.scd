(
s.reboot;
s.options.numOutputBusChannels = 2;
s.options.sampleRate = 44100;


/*ServerOptions.devices;

s.waitForBoot {
    "Input Channels: ".post; s.options.numInputBusChannels.postln;
    "Output Channels: ".post; s.options.numOutputBusChannels.postln;
};*/

s.meter;
v = s.volume;
v.min;
v.max;
v.volume = 0;
v.setVolumeRange(-10, 10);
v.mute;
v.unmute;

v.gui;

)


(
~baseLayerBuffer = Buffer.read(s, thisProcess.nowExecutingPath.dirname +/+ "/media/stems/base.wav");

~sampleLayerBuffer = Buffer.read(s, thisProcess.nowExecutingPath.dirname +/+ "media/stems/sample.wav");
)

(

SynthDef(\baseLayer, {
	 | buf, out = 0, volume = 1 |

    var sound;

    sound = PlayBuf.ar(2, buf, loop: 1);

    Out.ar(out, sound * volume);


}).add;

SynthDef(\sampleLayer, {
	 | buf, out = 0, volume = 0, pitchRatio = 1|

    var sound, finalSound;

    // Play the buffer
    sound = PlayBuf.ar(2, buf, loop: 1);

	finalSound = PitchShift.ar(sound,0.2, pitchRatio, 0.0, 0.0, 1.0, 0.0);


    // Output to specified channel
    Out.ar(out, finalSound * volume * 1.2); // Scale down the output level


}).add;

SynthDef(\sawtooth, {
	 | buf, out = 0, volume = 0, filterFreq = 1000|

    var sound, reverb;



	sound = RLPF.ar(Saw.ar(220,0.1), filterFreq, 0.2) *1.2;


    reverb = FreeVerb.ar(sound, 0.5, 0.5, 0.3);

    // Output to specified channel
    Out.ar(out, reverb * volume * 2); // Scale down the output level


}).add;
)

/*
INSTANTIATE SYNTHDEFS
*/
(
~baseLayerSynths = [0, 2].collect { |i| Synth(\baseLayer, [\buf, ~baseLayerBuffer,\out, i,])};

// Create multiple Synth instances for a cool effect
~sampleLayerSynths = [0, 2].collect { |i| Synth(\sampleLayer, [\buf, ~sampleLayerBuffer,\out, i,])};

~sawtoothSynths = [0, 2].collect { |i| Synth(\sawtooth, [\buf,\out, i,])};

~normalize = { |val, start, finish|
    // Ensure that the input 'val' is within 0 to 127.
    // Map it linearly from 0–127 to the range defined by 'start' and 'finish'.
    start + ((val / 127) * (finish - start))
};

)




/*
DECLARE AND INSTANTIATE CONTROLS
*/
/*(

//feedback, damping

n=EZSlider(nil, 300@42,"  pitchShift pitchRatio  ", ControlSpec.new(0.5, 1.5, step:0.1));

n.action_({ |ez| ~sampleLayerSynths.do { |synth| synth.set(\pitchRatio, ez.value); }});


m=EZSlider(nil, 300@42,"  pitchShift filterFreq  ", ControlSpec.new(30, 1000, step:10));

m.action_({ |ez| ~sawtoothSynths.do { |synth| synth.set(\filterFreq, ez.value); }});

o=EZSlider(nil, 300@42,"  sample off/on  ", ControlSpec.new(0, 1, step:1));

o.action_({ |ez|
	~sampleLayerSynths.do { |synth| synth.set(\volume, ez.value)};
	~sawtoothSynths.do { |synth| synth.set(\volume, ez.value)};


});



)*/

(
MIDIClient.init; // Initialize the MIDI system
MIDIIn.connectAll;


// Callback for Control Change messages
MIDIFunc.cc({ |src, chan, num, val|

	if(chan == 1){
		~normalize.value(src, 0.5, 1.5).postln;  // prints approximately 0.5039
		~sampleLayerSynths.do { |synth| synth.set(\pitchRatio, ~normalize.value(src, 0.5, 1.5)); }
	};

	if(chan == 2){
		~normalize.value(src, 300, 20000).postln;  // prints approximately 0.5039
		~sawtoothSynths.do { |synth| synth.set(\filterFreq, ~normalize.value(src, 300, 1000)); }	};

	/*
    ("CC - Source: " ++ src ++ ", Channel: " ++ chan ++
     ", Control: " ++ num ++ ", Value: " ++ val).postln;*/

});

// Callback for Note On messages
MIDIFunc.noteOn({ |src, chan, note, vel|
	"hand found".postln;
	~sampleLayerSynths.do { |synth| synth.set(\volume, 1)};
	~sawtoothSynths.do { |synth| synth.set(\volume, 1)};


    /*("Note On - Source: " ++ src ++ ", Channel: " ++ chan ++
     ", Note: " ++ note ++ ", Velocity: " ++ vel).postln;*/
});

// Callback for Note Off messages
MIDIFunc.noteOff({ |src, chan, note, vel|
	"hand missing".postln;
	~sampleLayerSynths.do { |synth| synth.set(\volume, 0)};
	~sawtoothSynths.do { |synth| synth.set(\volume, 0)};


    /*("Note Off - Source: " ++ src ++ ", Channel: " ++ chan ++
     ", Note: " ++ note ++ ", Velocity: " ++ vel).postln;*/
});
)

MIDIIn.disc;

