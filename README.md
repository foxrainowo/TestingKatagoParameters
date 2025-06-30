## TestingKatagoParameters
This is a database comparing the strength and affect of different parameter settings in KataGo. You can find the [Test_Result](https://github.com/foxrainowo/TestingKatagoParameters/releases) and [Latest_Test_Result](https://github.com/foxrainowo/TestingKatagoParameters/releases/latest). If you're interested in the strength of different KataGo networks and unofficial networks, you may want to check out [TestingKatagoNetworks](https://github.com/foxrainowo/TestingKatagoNetworks).

### Elo Estimation Tool
Download [summarize_elo.py](https://github.com/foxrainowo/TestingKatagoParameters/blob/main/summarize_elo.py) and run a command. Thanks to [@lightvector](https://github.com/lightvector) for developing this script, and I have made some improvements to it.

*You must install Python, then install the required dependencies: `pip install numpy scipy sgfmill colorama`*

**Normal Usage:**

	py summarize_elo.py "path" -standard {number} -prior {number} -create {v/p/t} -color -first-advantage -recursive
 
**For Instructions:**

	py summarize_elo.py -h
 
### How to contribute
I would collect *Game Records(SGFs)* and *Networks(bin.gz)* from public channels or You can send it to me.

**Standard requirements:**
- Fix a single board size, e.g., 19×19, 9×9.
- Fix the number of playouts **more than 50**.
- Fix a single game rule.
- More than 1000 games with each opponent.
