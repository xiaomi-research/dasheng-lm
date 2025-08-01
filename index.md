## Introduction
xxx integrates the powerful Dasheng audio encoder with the Qwen2.5-Omni-7B Thinker decoder through a unique caption-based alignment strategy.
Unlike conventional ASR-driven approaches, our model leverages general audio captions to capture comprehensive audio representations encompassing speech, environmental sounds, and musical elements
in a unified textual format. 
This design enables holistic audio understanding while maintaining exceptional computational efficiency.  
There two tasks that we want to show you some examples.
## Speech Meta Analysis(SMA)
### Speaker Timbre Analysis
This task illustrates the effectiveness of a speaker's timbre analysis via the following prompt:
```text
Qwen2.5-Omni：Listen to the provided audio and produce a caption describing the speaker's timbre.  
xxx-7B: Write an audio caption describing the speaker's timbre.
```
<section class="hero">
    <table style="text-align: center;">
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 1</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A female voice speaking. </p><p style="text-align: left;"><b>xxx-7B</b>: A female voice with a slightly high pitch and moderate volume delivers an enthusiastic monologue.</p><p style="text-align: center;"><audio controls><source src="data/audio/lqdBmAaxO78_219_90800000000002_229_908.flac" type="audio/flac"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 2</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A male in his twenties speaks in Spanish. </p><p style="text-align: left;"><b>xxx-7B</b>: A male speaker with a neutral tone delivers information in Spanish.</p><p style="text-align: center;"><audio controls><source src="data/audio/fiNsI10rlws_213_467_223_467.flac" type="audio/flac"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 3</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A female is speaking in a neutral mood. </p><p style="text-align: left;"><b>xxx-7B</b>: A female speaker with a neutral tone delivers a Portuguese monologue.</p><p style="text-align: center;"><audio controls><source src="data/audio/eZNWN8wueV8_21_4264_31_4264.flac" type="audio/flac"></audio></p></td>
        </tr>
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 4</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A female with a Spanish accent is speaking.</p><p style="text-align: left;"><b>xxx-7B</b>: A male speaker with a neutral tone delivers a Spanish-language monologue.</p><p style="text-align: center;"><audio controls><source src="data/audio/HelMeXcLA9A_34_272549999999995_44_2725.flac" type="audio/flac"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 5</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A man is speaking and music is playing.</p><p style="text-align: left;"><b>xxx-7B</b>: A male voice with a neutral tone delivers English pronunciation examples, including 'ring', 'wing', and 'running'.</p><p style="text-align: center;"><audio controls><source src="data/audio/UyVwZCyEufY_45_44_55_44.flac" type="audio/flac"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 6</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A woman is speaking.</p><p style="text-align: left;"><b>xxx-7B</b>: A female voice is heard, delivering a neutral-toned speech in English.</p><p style="text-align: center;"><audio controls><source src="data/audio/f0C7iAzA0Nc_89_23315_99_2331.wav" type="audio/wav"></audio></p></td>
        </tr>
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 7</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A man is speaking.</p><p style="text-align: left;"><b>xxx-7B</b>: A male voice with a deep, resonant tone delivers a monologue in English.</p><p style="text-align: center;"><audio controls><source src="data/audio/eQ0ohbOsZ2Y_252_66000000000003_262_66.flac" type="audio/flac"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 8</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A male voice is speaking.</p><p style="text-align: left;"><b>xxx-7B</b>: A male voice with a deep, resonant tone delivers a solemn monologue.</p><p style="text-align: center;"><audio controls><source src="data/audio/HDlbAuqr10A_85_0345_95_0345.flac" type="audio/flac"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 9</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A male voice speaking in a neutral tone.</p><p style="text-align: left;"><b>xxx-7B</b>: A male speaker with a neutral tone delivers a Portuguese message about toy donations.</p><p style="text-align: center;"><audio controls><source src="data/audio/elPijwMmVsY_14_753050000000002_24_7531.flac" type="audio/flac"></audio></p></td>
        </tr>
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 10</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A male voice speaking in English with a neutral mood.</p><p style="text-align: left;"><b>xxx-7B</b>: A male speaker with a neutral tone and clear enunciation discusses real-time data visualization.</p><p style="text-align: center;"><audio controls><source src="data/audio/mqTYKiVuEnU_43_33335_53_3334.flac" type="audio/flac"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 11</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: The speaker's voice is characterized by a mature, experienced tone, suggesting a mature adult male.</p><p style="text-align: left;"><b>xxx-7B</b>: A female speaker with a clear, articulate Telugu voice delivers a monologue.</p><p style="text-align: center;"><audio controls><source src="data/audio/39oK7RKDMQE_3_5600000000000005_13_56.flac" type="audio/flac"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 12</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: A female voice is speaking in English, with a neutral mood.</p><p style="text-align: left;"><b>xxx-7B</b>: A female voice with a neutral tone and clear enunciation.</p><p style="text-align: center;"><audio controls><source src="data/audio/lrMKnLcOHvk_344_3665_354_366.flac" type="audio/flac"></audio></p></td>
        </tr>
    </table>  
</section>

### Speaker language Analysis
This task illustrates the effectiveness of a speaker's language analysis via the following prompt: 
```text
请描述说话人的语言特性, 包括说话人的语种, 口音等.
```
<section class="hero">
    <table style="text-align: center;">
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 1</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: 说话人使用的是英语，带有斯里兰卡的口音。</p><p style="text-align: left;"><b>xxx-7B</b>: 说话人使用的是英语，带有印度口音。</p><p style="text-align: center;"><audio controls><source src="data/audio/common_voice_en_36758510.mp3" type="audio/mp3"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 2</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: 英语，美国口音</p><p style="text-align: left;"><b>xxx-7B</b>: 说话人使用的是英语，并带有美国口音。</p><p style="text-align: center;"><audio controls><source src="data/audio/common_voice_en_42452.mp3" type="audio/mp3"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 3</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: Benison is his estranged father's name and Fortune is his mother's maiden name.</p><p style="text-align: left;"><b>xxx-7B</b>: 说话人使用的是英语，具有美国口音。</p><p style="text-align: center;"><audio controls><source src="data/audio/common_voice_en_20210516.mp3" type="audio/wav"></audio></p></td>
        </tr>
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 4</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: 普通话-南京口音</p><p style="text-align: left;"><b>xxx-7B</b>: 说话人使用的是普通话，没有明显的口音。</p><p style="text-align: center;"><audio controls><source src="data/audio/test_3723.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 5</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: 普通话-女性口音</p><p style="text-align: left;"><b>xxx-7B</b>: 说话人使用的是普通话，没有明显的口音。</p><p style="text-align: center;"><audio controls><source src="data/audio/test_108.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 6</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: 普通话-北京口音</p><p style="text-align: left;"><b>xxx-7B</b>: 说话人使用的是普通话，没有明显的口音。</p><p style="text-align: center;"><audio controls><source src="data/audio/test_3977.wav" type="audio/wav"></audio></p></td>
        </tr>
    </table>  
</section>

## Sound Sphere Insight(SSI)
### Environmental Sound Recognition(Multi Label)
This task illustrates the effectiveness of environmental sound recognition via the following prompt: 
```text
Qwen2.5-Omni: Classify the given multi-label audio in English.
xxx-7B: Which labels describe the sound?
```
<section class="hero">
    <table style="text-align: center;">
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 1</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: {'Rattle'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Coin (dropping)', 'Domestic sounds, home sounds'}</p><p style="text-align: left;"><b>Label</b>: {'Coin (dropping)', 'Domestic sounds and home sounds'}</p><p style="text-align: center;"><audio controls><source src="data/audio/360908.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 2</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: {'Music'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Vehicle', 'Alarm', 'Vehicle horn, car horn, honking', 'Car', 'Motorcycle'}</p><p style="text-align: left;"><b>Label</b>: {'Vehicle horn and car horn and honking', 'Vehicle', 'Alarm', 'Car', 'Motorcycle'}</p><p style="text-align: center;"><audio controls><source src="data/audio/371450.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 3</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: {'Splash - splatter'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Water', 'Liquid', 'Splash, splatter'}</p><p style="text-align: left;"><b>Label</b>: {'Water', 'Liquid', 'Splash and splatter'}</p><p style="text-align: center;"><audio controls><source src="data/audio/83740.wav" type="audio/wav"></audio></p></td>
        </tr>
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 4</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: {'Radio'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Speech', 'Male speech, man speaking', 'Human voice'}</p><p style="text-align: left;"><b>Label</b>: {'Speech', 'Male speech and man speaking', 'Human voice'}</p><p style="text-align: center;"><audio controls><source src="data/audio/170667.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 5</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: {'Squeak'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Slam', 'Domestic sounds, home sounds', 'Door', 'Squeak'}</p><p style="text-align: left;"><b>Label</b>: {'Domestic sounds and home sounds', 'Slam', 'Door', 'Squeak'}</p><p style="text-align: center;"><audio controls><source src="data/audio/31768.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 6</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: {'Gunshot - gunfire'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Gunshot, gunfire', 'Explosion'}</p><p style="text-align: left;"><b>Label</b>: {'Explosion', 'Gunshot and gunfire'}</p><p style="text-align: center;"><audio controls><source src="data/audio/107619.wav" type="audio/wav"></audio></p></td>
        </tr>
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 7</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: {'Dog', 'bark'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Dog', 'Domestic animals, pets', 'Bark', 'Animal'}</p><p style="text-align: left;"><b>Label</b>: {'Dog', 'Bark', 'Domestic animals and pets', 'Animal'}</p><p style="text-align: center;"><audio controls><source src="data/audio/58382.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 8</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: {'Cat', 'meow'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Domestic animals, pets', 'Meow', 'Cat', 'Animal'}</p><p style="text-align: left;"><b>Label</b>: {'Meow', 'Cat', 'Domestic animals and pets', 'Animal'}</p><p style="text-align: center;"><audio controls><source src="data/audio/120017.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 9</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>:{'Bird'}</p><p style="text-align: left;"><b>xxx-7B</b>: {'Bird', 'Chirp, tweet', 'Bird vocalization, bird call, bird song', 'Wild animals', 'Animal'}</p><p style="text-align: left;"><b>Label</b>: {'Bird', 'Bird vocalization and bird call and bird song', 'Wild animals', 'Animal', 'Chirp and tweet'}</p><p style="text-align: center;"><audio controls><source src="data/audio/366857.wav" type="audio/wav"></audio></p></td>
        </tr>
    </table>  
</section>

### Music Instrument Recognition(Single Label)
This task illustrates the effectiveness of music instrument recognition via the following prompt: 
```text
Qwen2.5-Omni: Recognize the music instrument with keywords in English.
xxx-7B: What's the music instrument?
```
<section class="hero">
    <table style="text-align: center;">
        <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 1</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: violin</p><p style="text-align: left;"><b>xxx-7B</b>: string</p><p style="text-align: left;"><b>Label</b>: string</p><p style="text-align: center;"><audio controls><source src="data/audio/string_acoustic_056-047-075.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 2</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: keyboard</p><p style="text-align: left;"><b>xxx-7B</b>: guitar</p><p style="text-align: left;"><b>Label</b>: guitar</p><p style="text-align: center;"><audio controls><source src="data/audio/guitar_acoustic_030-102-127.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 3</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: organ</p><p style="text-align: left;"><b>xxx-7B</b>: vocal</p><p style="text-align: left;"><b>Label</b>: vocal</p><p style="text-align: center;"><audio controls><source src="data/audio/vocal_acoustic_000-057-025.wav" type="audio/wav"></audio></p></td>
        </tr>
         <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 4</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: reed</p><p style="text-align: left;"><b>xxx-7B</b>: flute</p><p style="text-align: left;"><b>Label</b>: flute</p><p style="text-align: center;"><audio controls><source src="data/audio/flute_synthetic_000-051-100.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 5</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: guitar</p><p style="text-align: left;"><b>xxx-7B</b>: keyboard</p><p style="text-align: left;"><b>Label</b>: keyboard</p><p style="text-align: center;"><audio controls><source src="data/audio/keyboard_electronic_069-074-075.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 6</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: keyboard</p><p style="text-align: left;"><b>xxx-7B</b>: mallet</p><p style="text-align: left;"><b>Label</b>: mallet</p><p style="text-align: center;"><audio controls><source src="data/audio/mallet_acoustic_047-103-075.wav" type="audio/wav"></audio></p></td>
        </tr>
         <tr>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 7</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: bass</p><p style="text-align: left;"><b>xxx-7B</b>: reed</p><p style="text-align: left;"><b>Label</b>: reed</p><p style="text-align: center;"><audio controls><source src="data/audio/reed_acoustic_023-039-127.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 8</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: guitar</p><p style="text-align: left;"><b>xxx-7B</b>: bass</p><p style="text-align: left;"><b>Label</b>: bass</p><p style="text-align: center;"><audio controls><source src="data/audio/bass_synthetic_135-067-025.wav" type="audio/wav"></audio></p></td>
            <td style="text-align: center; vertical-align: top;"><p><b>Example 9</b></p><p style="text-align: left;"><b>Qwen2.5-Omni</b>: bass</p><p style="text-align: left;"><b>xxx-7B</b>: organ</p><p style="text-align: left;"><b>Label</b>: organ</p><p style="text-align: center;"><audio controls><source src="data/audio/organ_electronic_113-026-025.wav" type="audio/wav"></audio></p></td>
        </tr>
    </table>  
</section>



## Citation
xxx is under the Apache License 2.0, and we encourage its use in **both research and business applications**.

If you find xxx useful in your research, please consider citing our work:

```bib
@misc{xxx7b,
    title = {xxx: Efficient Audio Understanding with General Audio Captions},
    author = {Xiaomi MiLM Plus Horizon Team},
    year = {2025},
}

```