# -*- coding: utf-8 -*-
"""
@Time    : 25/06/18 09:18
@Author  : XJH
"""

import pygame
from itertools import cycle
import random
import sys
import numpy as np


class GameEnv:
    def __init__(self):
        self.FPS = 30
        self.SCREENWIDTH = 288
        self.SCREENHEIGHT = 512
        self.BASEY = self.SCREENHEIGHT * 0.79
        self.PIPEGAPSIZE = 100
        self.PLAYERS_LIST = {}                  # 小鸟图片
        self.BACKGROUNDS_LIST = {}              # 背景图片
        self.PIPE_LIST = {}                     # 管道图片
        self.IMAGES = {}                        # 图片
        self.SOUNDS = {}                        # 声音素材
        self.HITMASKS = {}                      # 图片中每个像素点的透明度

        self.playerx, self.playery = 0., 0.     # 小鸟坐标
        self.score = 0                          # 得分
        self.playerIndex = 0                    # 小鸟状态（三状态）

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY = -9  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerRot = 45  # player's rotation
        self.playerVelRot = 3  # angular speed
        self.playerRotThr = 20  # rotation threshold
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        pass

    def render(self):
        pass

    def reset(self):
        """
        环境重置
        :return:
        """
        # 获取背景
        randBg = random.choice(self.BACKGROUNDS_LIST.keys())
        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[randBg]).convert()
        # 获取小鸟颜色
        randPlayer = random.choice(self.PLAYERS_LIST.keys())
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )
        # 获取管道颜色
        randpipe = random.randint(self.PIPE_LIST.keys())
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(self.PIPE_LIST[randpipe]).convert_alpha(), 180),
            pygame.image.load(self.PIPE_LIST[randpipe]).convert_alpha(),
        )
        # hismask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )
        # 生成两个新的管道
        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()
        # list of upper pipes
        upperPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        # list of lowerpipe
        lowerPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]
        self.score = self.playerIndex = loopIter = 0
        playerIndex = 0
        self.playerx, self.playery = int(self.SCREENWIDTH * 0.2), \
            int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)
        pipeVelX = -4
        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY = -9  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerRot = 45  # player's rotation
        self.playerVelRot = 3  # angular speed
        self.playerRotThr = 20  # rotation threshold
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps


    def step(self, action):
        if np.argmax(action) == 1:
            self.playerVelY = self.playerFlapAcc
            self.playerFlapped = True

        # check for crash here
        crashTest = self.checkCrash()

    def loadImage(self):
        """
        读取图像素材
        :return:
        """
        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS_LIST = {
            'red':
                (
                    'assets/sprites/redbird-upflap.png',
                    'assets/sprites/redbird-midflap.png',
                    'assets/sprites/redbird-downflap.png',
                ),
            'blue':
                (
                    'assets/sprites/bluebird-upflap.png',
                    'assets/sprites/bluebird-midflap.png',
                    'assets/sprites/bluebird-downflap.png',
                ),
            'yellow':
                (
                    'assets/sprites/yellowbird-upflap.png',
                    'assets/sprites/yellowbird-midflap.png',
                    'assets/sprites/yellowbird-downflap.png',
                )
        }
        self.BACKGROUNDS_LIST = {
            'day': 'assets/sprites/background-day.png',
            'night': 'assets/sprites/background-night.png',
        }
        self.PIPE_LIST = {
            'green': 'assets/sprites/pipe-green.png',
            'red': 'assets/sprites/pipe-red.png',
        }
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )
        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    def loadSOund(self):
        """
        读取声音素材
        :return:
        """
        if 'win' in sys.platform:
            soundExt = '.wav'
        else:
            soundExt = '.ogg'

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE},  # lower pipe
        ]

    def checkCrash(self, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        playerw = self.IMAGES['player'][0].get_width()
        playerh = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if self.playery + self.playerx >= self.BASEY - 1:
            return [True, True]
        else:

            playerRect = pygame.Rect(self.playerx, self.playery,
                                     playerw, playerh)
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][self.playerIndex]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False