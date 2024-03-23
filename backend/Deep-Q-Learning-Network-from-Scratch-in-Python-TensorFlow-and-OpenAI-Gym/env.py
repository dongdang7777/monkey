import sys
from pathlib import Path
import random
import importlib
import json
from gym import Env
sys.path.append(str(Path("GameManager.py").parent))
from Game.test import *
from Game.playerActions import defense_actions, attack_actions, projectile_actions, nullDef, nullAtk, nullProj
from Game.gameSettings import *
from Game.Skills import *
from Game.projectiles import *
from Game.turnUpdates import *
from Game.PlayerConfigs import Player_Controller
from ScriptingHelp.usefulFunctions import *
from gym.spaces import Discrete
from math import sqrt
# Manually choose bot files to test
SUBMISSIONPATH = "Submissions"
PATH1 = "Bot1"
PATH2 = "Bot2"
import numpy as np

# Get scripts from bot files and return as script objects
def getPlayerFiles(path1, path2, subpath):
    submission_files = Path(subpath)
    p1module = submission_files / (path1 + ".py")
    p2module = submission_files / (path2 + ".py")
    if p1module.is_file() and p2module.is_file():
        # Ensures path works on mac and windows
        subpath = subpath.replace('\\', '.')
        subpath = subpath.replace('/', '.')
        p1 = importlib.import_module(subpath + "." + path1)
        p2 = importlib.import_module(subpath+ "." + path2)
        return p1, p2
    else:
        raise Exception("A file does not exist in " + subpath)


# Checks for players moving into each other
def checkCollision(player1, player2, knock1, knock2, check_midair = False):
    if (correct_dir_pos(player1, player2, knock1, knock2)):
        # If an overlap occured, then a collision has occured, so set
        # horizontal midair velocity to 0
        player1._velocity = 0
        player2._velocity = 0
    elif check_midair:
        # Check for midair players moving towards each other
        # If they end up face-to-face midair, set horizontal velocity to 0
        if ((player1._yCoord == player2._yCoord) and 
            (abs(player1._xCoord - player2._xCoord) == 1)
            and (player1._direction != player2._direction)):
            player1._velocity = 0
            player2._velocity = 0
                
# Plays out a single turn, doesn't check deaths
def executeOneTurn(action,player1, player2, p1_script, p2_script, p1_json_dict, p2_json_dict, projectiles):
    # Initializing knockbacks: knock1 = knockback INFLICTED by player1 on player 2
    knock1 = knock2 = 0
    stun1 = stun2 = 0
    # If midair, start falling/rising and check if a collision occurs
    updateMidair(player1)
    checkCollision(player1, player2, knock1, knock2)
    updateMidair(player2)
    checkCollision(player1, player2, knock1, knock2)


    # Check for existing projectiles belonging to each player
    p1_projectiles = [proj["projectile"] for proj in projectiles if proj["projectile"]._player._id == 1]
    p2_projectiles = [proj["projectile"] for proj in projectiles if proj["projectile"]._player._id == 2]
    
    # Pass relevant information to player scripts, and get a move from them
    p1_move = action;
    p2_move = p2_script.get_move(player2, player1, p2_projectiles, p1_projectiles)
  
    # In case the scripts return None
    if not p1_move:
        p1_move = ("NoMove",)
    if not p2_move:
        p2_move = ("NoMove",)
        
    # Add their move to their list of inputs
    player1._inputs.append(p1_move)
    player2._inputs.append(p2_move)
    
    # Get move from input list
    act1 = player1._action()
    act2 = player2._action()
    
    # Get game information from the result of the players performing their inputs
    knock1, stun1, knock2, stun2, projectiles = performActions(player1, player2, 
                                        act1, act2, stun1, stun2, 
                                        projectiles)
    # JSONFILL always True now...
    # Writes to json files the current actions, positions, hp etc...
    if JSONFILL:
        playerToJson(player1, p1_json_dict, not JSONFILL)
        playerToJson(player2,p2_json_dict, not JSONFILL)
        
    # Check if players move into each other, correct it if they do
    checkCollision(player1, player2, knock1, knock2)
    
    # Make any currently existing projectiles move, and record them in json files
    projectiles, knock1, stun1, knock2, stun2 = projectile_move(projectiles, 
                            knock1, stun1, knock2, stun2, player1, player2,
                            p1_json_dict, p2_json_dict)


    # Only determine knockback and stun after attacks hit
    if (knock1 or stun1) and not player2._superarmor:
        player2._xCoord += knock1
        if not player2._stun:
            player2._stun = stun1
    if (knock2 or stun2) and not player1._superarmor:
        player1._xCoord += knock2
        if not player1._stun:
            player1._stun = stun2
        
    # Final position correction, if any, due to projectiles      
    checkCollision(player1, player2, knock1, knock2, True)
        
    updateCooldown(player1)
    updateCooldown(player2)
    
    updateBuffs(player1)
    updateBuffs(player2)
    
    p1_dead = checkDeath(player1)
    p2_dead = checkDeath(player2)

    # Second write to json files, for any movement due to projectiles, and to 
    # check if a player got hurt
    playerToJson(player1, p1_json_dict, fill=JSONFILL, checkHurt = JSONFILL)
    playerToJson(player2,p2_json_dict, fill=JSONFILL, checkHurt = JSONFILL)

    return projectiles, p1_dead, p2_dead

def setupGame(p1_script, p2_script, leftstart=LEFTSTART, rightstart=RIGHTSTART):
    
    # Initializes player scripts as player controller objects
    player1 = Player_Controller(leftstart,0,HP,GORIGHT, *p1_script.init_player_skills(), 1)
    player2 = Player_Controller(rightstart,0,HP,GOLEFT, *p2_script.init_player_skills(), 2)
    # Ensure that valid primary and secondary skills are set
    assert(check_valid_skills(*p1_script.init_player_skills()))
    assert(check_valid_skills(*p2_script.init_player_skills()))
    return player1,player2
    
# Resets player shield strength
def resetBlock(player):
    player._block._regenShield()
    player._blocking = False
    
# Carries out player actions, return any resulting after effects to main loop  
def performActions(player1, player2, act1, act2, stun1, stun2, projectiles):
    knock1 = knock2 = 0

    # Empty move if player is currently stunned or doing recovery ticks
    if player1._stun or player1._recovery:
        act1 = ("NoMove", "NoMove")
        updateStun(player1)
    if player2._stun or player2._recovery:
        act2 = ("NoMove", "NoMove")
        updateStun(player2)
    
    # Checks if player does something to cancel a skill
    if player1._mid_startup or player1._skill_state:
        if player1._inputs[-1][0] in ("move", "block"):
            player1._skill_state = False
            player1._mid_startup = False
        else:
            act1 = player1._moves[-1]
            
    if player2._mid_startup or player2._skill_state:
        if player2._inputs[-1][0] in ("move", "block"):
            player2._skill_state = False
            player2._mid_startup = False
        else:
            act2 = player2._moves[-1]
        
    # Check if no valid move is input, or if the player is recovering 
    # If so, set act to None to prevent further checks
    if act1[0] not in (attack_actions.keys() | defense_actions.keys() | projectile_actions.keys()):
        if player1._recovery:
            player1._moves.append(("recover", None))
            updateRecovery(player1)
        else:
            player1._moves.append(("NoMove", "NoMove"))
        resetBlock(player1)
        act1 = None
    if act2[0] not in (attack_actions.keys() | defense_actions.keys() | projectile_actions.keys()):
        if player2._recovery:
            player2._moves.append(("recover", None))
            updateRecovery(player2)
        else:
            player2._moves.append(("NoMove", "NoMove"))
        resetBlock(player2)
        act2 = None

    # nullDef, nullAtk, nullProj = default functions that return (0,0) or None
    # actions can only occur if the player is not stunned
    # if a defensive action is taken, it has priority over damage moves/skills
    # defensive = any skill that does not deal damage
    
    # Movements are cached, and then carried out based on position 
    # If there are movements, set act to None to prevent going into attack check
    cached_move_1 = cached_move_2 = None
    if act1:
        if act1[0] != "block":
            resetBlock(player1)
        cached_move_1 = defense_actions.get(act1[0], nullDef)(player1, player2, act1)
        if cached_move_1:
            act1 = None
    if act2:
        if act2[0] != "block":
            resetBlock(player2)
        cached_move_2 = defense_actions.get(act2[0], nullDef)(player2, player1, act2)
        if cached_move_2:
            act2 = None
    # Prevent players that are directly facing each other from moving into each other
    if isinstance(cached_move_1, list) and isinstance(cached_move_2, list):
        if (check_move_collision(player1, player2, cached_move_1, cached_move_2) 
            and cached_move_1[1] == cached_move_2[1] and 
            abs(player1._xCoord - player2._xCoord) == 1):
            cached_move_1 = cached_move_2 = None
            player1._moves[-1] = ("NoMove", None)
            player2._moves[-1] = ("NoMove", None) 
    
    # Further checks for valid movement
    # Prevent horizontal movement if it would result in moving into a still player
    # Diagonal movements are allowed, since midair collision checks occur after
    if isinstance(cached_move_1, list):
        if player1._xCoord + cached_move_1[0] == player2._xCoord and cached_move_2 in ([0,0], None) and not cached_move_1[1]:
            cached_move_1[0] = 0
        player1._xCoord += cached_move_1[0]
        player1._yCoord += cached_move_1[1]
        player1._moves[-1] = ("move", (cached_move_1[0]*player1._direction, cached_move_1[1]))
    if isinstance(cached_move_2, list):
        if player2._xCoord + cached_move_2[0] == player1._xCoord and cached_move_1 in ([0,0], None) and not cached_move_2[1]:
            cached_move_2[0] = 0
        player2._xCoord += cached_move_2[0]
        player2._yCoord += cached_move_2[1]
        player2._moves[-1] = ("move", (cached_move_2[0]*player2._direction, cached_move_2[1]))
        
    # Prevent from going offscreen
    correctPos(player1)
    correctPos(player2)

    # Now check for damage dealing actions
    # Get any knockback and stun values if an attack lands
    # Summon projectiles if any projectile skills were casted
    if act1:
        knock1, stun1 = attack_actions.get(act1[0], nullAtk)(player1, player2, act1)
        proj_obj = projectile_actions.get(act1[0], nullProj)(player1, player2, act1)
        if proj_obj:
            projectiles.append(proj_obj)
        resetBlock(player1)
    if act2:
        knock2, stun2 = attack_actions.get(act2[0], nullAtk)(player2, player1, act2)
        proj_obj = projectile_actions.get(act2[0], nullProj)(player2, player1, act2)
        if proj_obj:
            projectiles.append(proj_obj)
        resetBlock(player2)

    # Correct positioning again just in case
    correctPos(player1)
    correctPos(player2)
    
    # Move to next move in player input list
    player1._move_num += 1
    player2._move_num += 1
    
    return knock1, stun1, knock2, stun2, projectiles

# Initializes json object 
def get_empty_json():
    return {
        'hp': [],
        'xCoord': [],
        'yCoord': [],
        'state': [],
        'actionType': [],
        'stun': [],
        'midair': [],
        'falling':[],
        'direction':[],
        'ProjectileType': None,
        'projXCoord':[],
        'projYCoord':[]
    }
                              
# Main game loop            
    
        
    # Write into json files
def get_nearest_projectile(player, projectiles):
    nearest_projectile = None
    nearest_distance = float('inf')

    for proj in projectiles:
        player_x, player_y = get_pos(player)
        proj_x, proj_y = get_proj_pos(proj)
        distance = sqrt((player_x - proj_x)**2 + (player_y - proj_y)**2)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_projectile = proj

    return nearest_projectile

# Allows to run directly from GameManager to simulate single rounds
# if __name__ == "__main__":
class fighterEnv(Env):
    def __init__(self):
        PRIMARY_SKILL = DashAttackSkill
        SECONDARY_SKILL = Hadoken
        JUMP = ("move", (0,1))
        FORWARD = ("move", (1,0))
        BACK = ("move", (-1,0))
        JUMP_FORWARD = ("move", (1,1))
        JUMP_BACKWARD = ("move", (-1, 1))
        LIGHT = ("light",)
        HEAVY = ("heavy",)
        BLOCK = ("block",)
        PRIMARY = get_skill(PRIMARY_SKILL)
        SECONDARY = get_skill(SECONDARY_SKILL)
        NOMOVE = "NoMove"
        self.Actions=[PRIMARY,SECONDARY,JUMP,FORWARD,BACK,JUMP_FORWARD,JUMP_BACKWARD,LIGHT,HEAVY,BLOCK,NOMOVE]
        
        self.path1=PATH1;
        self.path2=PATH2;
        self.projectiles=[];
        self.p1_dead=False;
        self.p2_dead=False;
        self.submissionpath=SUBMISSIONPATH
        self.roundNum=1;
        self.p1, self.p2 = getPlayerFiles(self.path1, self.path2, self.submissionpath)
        self.p1_script = self.p1.Script()
        self.p2_script = self.p2.Script()
        self.player1, self.player2 = setupGame(self.p1_script, self.p2_script)
        # Check if file exists if so delete it 
        self.player_json = Path("jsonfiles/")
        # create new battle file with player jsons
        self.new_battle = self.player_json / f"Round_{self.roundNum}"
        self.player1_json = self.new_battle / "p1.json"
        self.player2_json = self.new_battle / "p2.json"
        # create round result file
        self.round_results_json = self.new_battle / "round.json"
        # get list of battles 
        files = self.player_json.glob("*")
        battles = [x for x in files if x.is_dir()]   
        # check if this battle has not happened before
        if f"Round {self.roundNum}" not in battles:
            self.player1_json.parent.mkdir(parents=True, exist_ok=True)
            self.player2_json.parent.mkdir(parents=True, exist_ok=True)
            self.round_results_json.parent.mkdir(parents=True, exist_ok=True)
            
        self.player1_json.open("w")
        self.player2_json.open("w")
        self.round_results_json.open("w")
        # structure the dict, no need to structure round result dict until the end
        self.p1_json_dict = get_empty_json()
        self.p2_json_dict = get_empty_json()
        
        # Initialize variables
        self.projectiles = []
        self.tick = 0
        self.max_tick = TIME_LIMIT * MOVES_PER_SECOND
        
        # Buffer turn : for smoothness
        for _ in range(BUFFERTURNS * 2): # 2 since fill ticks
            playerToJson(self.player1, self.p1_json_dict, fill=True, start=True)
            playerToJson(self.player2, self.p2_json_dict, fill=True, start=True)
            projectileToJson(None, self.p1_json_dict, False, fill=True)
            projectileToJson(None, self.p2_json_dict, False, fill=True)
            self.tick += 1
            self.max_tick += 1
            
        # Loops through turns
        
        # Write into json files
            # choose random player to win if tie

        

        # DO NOTHING
        # JUMP = ("move", (0,1))
        # FORWARD = ("move", (1,0))
        # BACK = ("move", (-1,0))
        # JUMP_FORWARD = ("move", (1,1))
        # JUMP_BACKWARD = ("move", (-1, 1))
        # LIGHT = ("light",)
        # HEAVY = ("heavy",)
        # BLOCK = ("block",)
        # Primary
        # Secondary
        

        self.action_space = Discrete(11)
        # Temperature y

        #Player x
        #Player y
        #Enemy x
        #Enemy y
        #Player HP
        #Enemy HP

        self.playerHP=get_hp(self.player1)
        self.enemyHP=get_hp(self.player2)
    
        self.p1_projectiles = [proj["projectile"] for proj in self.projectiles if proj["projectile"]._player._id == 1]
        self.p2_projectiles = [proj["projectile"] for proj in self.projectiles if proj["projectile"]._player._id == 2]
        self.playerX,self.playerY=get_pos(self.player1);
        self.enemyX,self.enemyY=get_pos(self.player2);
        
        self.playerStun=get_stun_duration(self.player1);
        self.enemyStun=get_stun_duration(self.player2);

        self.playerPrimary=get_primary_cooldown(self.player1);
        self.playerSecondary=get_secondary_cooldown(self.player1);
        self.enemyPrimary=get_primary_cooldown(self.player2);
        self.enemySecondary=get_secondary_cooldown(self.player2);
        self.playerHeavy=heavy_on_cooldown(self.player1);
        self.enemyHeavy=heavy_on_cooldown(self.player2);

        self.playerRecovery=get_recovery(self.player1);
        self.enemyRecovery=get_recovery(self.player2);

        self.playerBlock=get_block_status(self.player1);
        self.enemyBlock=get_block_status(self.player2);
        
        if (self.enemyHeavy==True):self.enemyHeavy=1;
        else: self.enemyHeavy=0
        if (self.playerHeavy==True):self.playerHeavy=1;
        else: self.playerHeavy=0
        

        proj=get_nearest_projectile(self.player1,self.p2_projectiles);
        self.projX=-1
        self.projY=-1
        if (proj!=None):
                self.projX,self.projY=get_proj_pos(proj)
                print("Projectile spawned ",self.projX,self.projY);
        

    

        self.state =[
            self.playerHP,
            self.enemyHP,
            self.playerX,
            self.playerY,
            self.enemyX,
            self.enemyY,
            self.playerStun,
            self.enemyStun,
            self.playerPrimary,
            self.enemyPrimary,
            self.playerSecondary,
            self.enemySecondary,
            self.playerHeavy,
            self.enemyHeavy,
            self.playerRecovery,
            self.enemyRecovery,
            self.playerBlock,
            self.enemyBlock,
            self.projX,
            self.projY
        ]
        self.state=np.array(self.state);
        # Set shower length
        
    def step(self, action):
        realAction=self.Actions[action];
        info = {}
        reward=0;
        self.projectiles, self.p1_dead, self.p2_dead = executeOneTurn(realAction,self.player1, 
            self.player2, self.p1_script, self.p2_script, self.p1_json_dict, self.p2_json_dict, 
            self.projectiles)
        self.tick+=1;
        
        done = not (not(self.p1_dead or self.p2_dead) and (self.tick < self.max_tick))
        
        prePlayerHealth=self.playerHP;
        preEnemyHealth=self.enemyHP;

        #fdji
        if (done):
            self.player1_json.write_text(json.dumps(self.p1_json_dict))
            self.player2_json.write_text(json.dumps(self.p2_json_dict))
            print_results = False
            if print_results:
                for key in self.p1_json_dict.keys():
                    print(key)
                    print(self.p1_json_dict[key])
                for key in self.p2_json_dict.keys():
                    print(key)
                    print(self.p2_json_dict[key])

                for json_key in self.p1_json_dict:
                    if json_key != "ProjectileType":
                        print(f"{json_key} : {len(self.p1_json_dict[json_key])}")
                        
                for json_key in self.p2_json_dict:
                    if json_key != "ProjectileType":
                        print(f"{json_key} : {len(self.p2_json_dict[json_key])}")
                        
                print(f"START BUFFERS: {BUFFERTURNS}, ACTUAL TURNS: {len(self.player1._inputs)}")
                print(f"jsonfill is {JSONFILL}")
                print(f"{self.path1} HP: {self.player1._hp} --  {self.path2} HP: {self.player2._hp}")
            
            winner = None
            
            if self.player1._hp > self.player2._hp:
                print(f"{self.path1} won in {self.tick} turns!")
                winner = self.path1
            elif self.player1._hp < self.player2._hp:
                print(f"{self.path2} won in {self.tick} turns!")
                winner = self.path2
            else:
                print('Tie!')
            
            round_info = {'p1': self.path1, 'p2':self.path2, 'winner':winner, 'roundNum':self.roundNum}
            self.round_results_json.write_text(json.dumps(round_info))
        


                #Player x
        #Player y
        #Enemy x
        #Enemy y
        #Player HP
        #Enemy HP

        self.playerHP=get_hp(self.player1)
        self.enemyHP=get_hp(self.player2)
    
        self.p1_projectiles = [proj["projectile"] for proj in self.projectiles if proj["projectile"]._player._id == 1]
        self.p2_projectiles = [proj["projectile"] for proj in self.projectiles if proj["projectile"]._player._id == 2]
        self.playerX,self.playerY=get_pos(self.player1);
        self.enemyX,self.enemyY=get_pos(self.player2);
        
        self.playerStun=get_stun_duration(self.player1);
        self.enemyStun=get_stun_duration(self.player2);

        self.playerPrimary=get_primary_cooldown(self.player1);
        self.playerSecondary=get_secondary_cooldown(self.player1);
        self.enemyPrimary=get_primary_cooldown(self.player2);
        self.enemySecondary=get_secondary_cooldown(self.player2);
        self.playerHeavy=heavy_on_cooldown(self.player1);
        self.enemyHeavy=heavy_on_cooldown(self.player2);

        self.playerRecovery=get_recovery(self.player1);
        self.enemyRecovery=get_recovery(self.player2);

        self.playerBlock=get_block_status(self.player1);
        self.enemyBlock=get_block_status(self.player2);
        
        if (self.enemyHeavy==True):self.enemyHeavy=1;
        else: self.enemyHeavy=0
        if (self.playerHeavy==True):self.playerHeavy=1;
        else: self.playerHeavy=0
        

        proj=get_nearest_projectile(self.player1,self.p2_projectiles);
        self.projX=-1
        self.projY=-1
        if (proj!=None):
                self.projX,self.projY=get_proj_pos(proj)
                print("Projectile spawned ",self.projX,self.projY);
        

        self.state =[
            self.playerHP,
            self.enemyHP,
            self.playerX,
            self.playerY,
            self.enemyX,
            self.enemyY,
            self.playerStun,
            self.enemyStun,
            self.playerPrimary,
            self.enemyPrimary,
            self.playerSecondary,
            self.enemySecondary,
            self.playerHeavy,
            self.enemyHeavy,
            self.playerRecovery,
            self.enemyRecovery,
            self.playerBlock,
            self.enemyBlock,
            self.projX,
            self.projY
        ]
        self.state=np.array(self.state);
        a=(self.playerHP-prePlayerHealth)*3;
        b=max(0,(preEnemyHealth-self.enemyHP)*2)
        reward+=a;
        reward+=b
        
        self.projectiles
        if (done):
            if (self.playerHP>self.enemyHP):reward+=200+self.max_tick-self.tick;
            elif (self.playerHP<self.enemyHP): reward-=(200+self.max_tick-self.tick);
        
        if (a>0):
            print("Rewarded for healing")
        elif (a<0):
            print("Punished for losing health")
        if (b>0):
            print("Rewarded for damaging")

        print("Player's health",self.playerHP);
        print("Enemy's health",self.enemyHP);
        reward+=self.enemyStun*2;
        reward-=self.playerStun*2;
        print(realAction);
        return self.state, reward, done, info

    def render(self):
        pass
    
    
    def reset(self):
        self.path1=PATH1;
        self.path2=PATH2;
        self.projectiles=[];
        self.p1_dead=False;
        self.p2_dead=False;
        self.submissionpath=SUBMISSIONPATH
        self.roundNum=1;
        self.p1, self.p2 = getPlayerFiles(self.path1, self.path2, self.submissionpath)
        self.p1_script = self.p1.Script()
        self.p2_script = self.p2.Script()
        self.player1, self.player2 = setupGame(self.p1_script,self.p2_script)
        # Check if file exists if so delete it 
        self.player_json = Path("jsonfiles/")
        # create new battle file with player jsons
        self.new_battle = self.player_json / f"Round_{self.roundNum}"
        self.player1_json = self.new_battle / "p1.json"
        self.player2_json = self.new_battle / "p2.json"
        # create round result file
        self.round_results_json = self.new_battle / "round.json"
        # get list of battles 
        files = self.player_json.glob("*")
        battles = [x for x in files if x.is_dir()]   
        # check if this battle has not happened before
        if f"Round {self.roundNum}" not in battles:
            self.player1_json.parent.mkdir(parents=True, exist_ok=True)
            self.player2_json.parent.mkdir(parents=True, exist_ok=True)
            self.round_results_json.parent.mkdir(parents=True, exist_ok=True)
            
        self.player1_json.open("w")
        self.player2_json.open("w")
        self.round_results_json.open("w")
        # structure the dict, no need to structure round result dict until the end
        self.p1_json_dict = get_empty_json()
        self.p2_json_dict = get_empty_json()
        
        # Initialize variables
        self.projectiles = []
        self.tick = 0
        self.max_tick = TIME_LIMIT * MOVES_PER_SECOND
        
        # Buffer turn : for smoothness
        for _ in range(BUFFERTURNS * 2): # 2 since fill ticks
            playerToJson(self.player1, self.p1_json_dict, fill=True, start=True)
            playerToJson(self.player2, self.p2_json_dict, fill=True, start=True)
            projectileToJson(None, self.p1_json_dict, False, fill=True)
            projectileToJson(None, self.p2_json_dict, False, fill=True)
            self.tick += 1
            self.max_tick += 1
            
            
        
      
        self.playerHP=get_hp(self.player1)
        self.enemyHP=get_hp(self.player2)
    
        self.p1_projectiles = [proj["projectile"] for proj in self.projectiles if proj["projectile"]._player._id == 1]
        self.p2_projectiles = [proj["projectile"] for proj in self.projectiles if proj["projectile"]._player._id == 2]
        self.playerX,self.playerY=get_pos(self.player1);
        self.enemyX,self.enemyY=get_pos(self.player2);
        
        self.playerStun=get_stun_duration(self.player1);
        self.enemyStun=get_stun_duration(self.player2);

        self.playerPrimary=get_primary_cooldown(self.player1);
        self.playerSecondary=get_secondary_cooldown(self.player1);
        self.enemyPrimary=get_primary_cooldown(self.player2);
        self.enemySecondary=get_secondary_cooldown(self.player2);
        self.playerHeavy=heavy_on_cooldown(self.player1);
        self.enemyHeavy=heavy_on_cooldown(self.player2);

        self.playerRecovery=get_recovery(self.player1);
        self.enemyRecovery=get_recovery(self.player2);

        self.playerBlock=get_block_status(self.player1);
        self.enemyBlock=get_block_status(self.player2);
        
        if (self.enemyHeavy==True):self.enemyHeavy=1;
        else: self.enemyHeavy=0
        if (self.playerHeavy==True):self.playerHeavy=1;
        else: self.playerHeavy=0
        

        proj=get_nearest_projectile(self.player1,self.p2_projectiles);
        self.projX=-1
        self.projY=-1
        if (proj!=None):
                self.projX,self.projY=get_proj_pos(proj)
                print("Projectile spawned ",self.projX,self.projY);
        

        self.state =[
            self.playerHP,
            self.enemyHP,
            self.playerX,
            self.playerY,
            self.enemyX,
            self.enemyY,
            self.playerStun,
            self.enemyStun,
            self.playerPrimary,
            self.enemyPrimary,
            self.playerSecondary,
            self.enemySecondary,
            self.playerHeavy,
            self.enemyHeavy,
            self.playerRecovery,
            self.enemyRecovery,
            self.playerBlock,
            self.enemyBlock,
            self.projX,
            self.projY
        ]
        self.state=np.array(self.state);
        return self.state

if __name__=="__main__":
    env=fighterEnv();
    episodes = 10
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            #env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))