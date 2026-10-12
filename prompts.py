"""Fixed songs (original lyrics) and style axes for the experiments."""

LYRICS = {
    "river": """[Intro]

[Verse]
Light on the water, light on the sea
Morning is calling, calling for me
Boots on the gravel, coffee in hand
Nobody told me, I had to stand

[Pre-Chorus]
And the wind keeps turning
The old map burning

[Chorus]
Run to the river, run to the light
Hold to the feeling, hold it tonight
We were the fire, we were the flame
Nothing between us will ever be the same

[Verse]
Paint on the ceiling, songs on the floor
Somebody knocking, open the door
Winter is over, summer is near
Everything's changing, everything's here

[Chorus]
Run to the river, run to the light
Hold to the feeling, hold it tonight
We were the fire, we were the flame
Nothing between us will ever be the same

[Bridge]
Oh, the city's sleeping
Oh, the secrets keeping

[Chorus]
Run to the river, run to the light
Hold to the feeling, hold it tonight
We were the fire, we were the flame
Nothing between us will ever be the same

[Outro]
""",
    "neon": """[Intro]

[Verse]
Neon on the windshield, rain across the glass
Every station's playing songs about the past
I keep driving slower so the night won't end
Counting all the exits, counting every friend

[Chorus]
Turn it up, turn it up, let the radio play
Turn it up, turn it up, till the dark goes away
We are the static, we are the sound
Spinning forever, never coming down

[Verse]
Coffee in the morning, sugar on the street
Somebody is dancing to a broken beat
Shadows on the corner, laughter in the hall
We were never strangers, never strangers at all

[Chorus]
Turn it up, turn it up, let the radio play
Turn it up, turn it up, till the dark goes away
We are the static, we are the sound
Spinning forever, never coming down

[Outro]
""",
    "short": """[Verse]
Light on the water, light on the sea
Morning is calling, calling for me

[Chorus]
Run to the river, run to the light
Hold to the feeling, hold it tonight
""",
}

LYRICS["corpus"] = """[Verse]
Paint on the ceiling, songs on the floor
Somebody knocking, open the door
Winter is over, summer is near
Everything's changing, everything's here

[Chorus]
Run to the river, run to the light
Hold to the feeling, hold it tonight
"""

# Exp 2 corpus: a base style per genre, and attribute poles appended to it.
# Each (genre, seed) renders both poles of every axis, so a pole difference is
# a paired contrast with everything else fixed.
GENRES = {
    "pop": "pop, female vocal, piano",
    "rock": "rock, male vocal, electric guitar",
    "folk": "folk, female vocal, acoustic guitar",
    "electro": "electronic, male vocal, synth",
}
AXES = {
    "energy": ("energetic, powerful, intense", "calm, gentle, soft"),
    "tempo": ("fast, 150 bpm", "slow, 70 bpm"),
    "density": ("full band, dense arrangement, layered", "sparse, minimal arrangement, solo"),
    "timbre": ("bright, distorted, aggressive", "warm, mellow, dark"),
    "vocal": ("vocal-forward, lead vocal", "instrumental"),
}
# Neutral prompts for steering verification (not in the corpus).
HELDOUT = {
    "soul": "soul, female vocal, Rhodes piano, bass",
    "country": "country, male vocal, acoustic guitar, fiddle",
}

SONGS = {
    "folk": dict(style="indie folk, warm acoustic guitar, female vocal, 92 bpm", lyrics=LYRICS["river"]),
    "synthpop": dict(style="synthwave, pop, male vocal, synth, drum machine, 118 bpm", lyrics=LYRICS["neon"]),
    "rock": dict(style="rock, electric guitar, powerful vocal, tight drums, 128 bpm", lyrics=LYRICS["neon"]),
}
