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
        self.actions = 2                        # 动作空间
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
        self.loopIter = 0                       # 小鸟三状态切换
        self.playerIndexGen = cycle([0, 1, 2, 1])   # 循环
        self.score = 0                          # 得分
        self.playerIndex = 0                    # 小鸟状态（三状态）

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY = 0  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerRot = 45  # player's rotation
        self.playerVelRot = 3  # angular speed
        self.playerRotThr = 20  # rotation threshold
        self.playerFlapAcc = -1  # players speed on flapping
        self.playerFlapped = False  # True when player flaps
        self.pipeVelX = -4          #

        self.upperPipes = []                    # 上管道坐标
        self.lowerPipes = []                    # 下管道坐标
        self.basex = 0                          # basex
        self.baseShift = 100

        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')
        # 读取素材
        self.loadImage()
        self.loadSound()

        pass

    def render(self):
        pygame.display.update()
        self.FPSCLOCK.tick(self.FPS)

    def reset(self):
        """
        环境重置
        :return:
        """
        # 获取背景
        randBg = random.choice(list(self.BACKGROUNDS_LIST.keys()))
        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[randBg]).convert()
        # 获取小鸟颜色
        randPlayer = random.choice(list(self.PLAYERS_LIST.keys()))
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )
        # 获取管道颜色
        randpipe = random.choice(list(self.PIPE_LIST.keys()))
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
        self.upperPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]
        self.score = self.playerIndex = loopIter = 0
        playerIndex = 0
        self.playerx, self.playery = int(self.SCREENWIDTH * 0.2), \
            int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)
        pipeVelX = -4
        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY = 0  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerRot = 0  # player's rotation
        self.playerVelRot = 3  # angular speed
        self.playerRotThr = 20  # rotation threshold
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps
        self.basex = 0
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data

    def step(self, action):
        reward = 0
        # print(action)
        # if np.argmax(action) == 1:
        if action == 1:
            self.playerVelY = self.playerFlapAcc            # 小鸟沿着Y轴方向的速度
            self.playerFlapped = True                       # 在flappy状态

        # check for score
        playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward += 10000
                # self.SOUNDS['point'].play()

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 45

        playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
        self.playery += min(self.playerVelY, self.BASEY - self.playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
        # print score so player overlaps the score
        self.showScore(self.score)

        # Player rotation has a threshold
        visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            visibleRot = self.playerRot

        playerSurface = pygame.transform.rotate(self.IMAGES['player'][self.playerIndex], visibleRot)
        self.SCREEN.blit(playerSurface, (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        # check for crash here
        crashTest = self.checkCrash(self.upperPipes, self.lowerPipes)
        if crashTest[0]:
            return image_data, reward, True

        return image_data, reward+1, False

        


    def loadImage(self):
        """
        读取图像素材
        :return:
        """
        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS_LIST = {
            'red':
                (
                    'assets/sprites/redbird-upflap.jpg',
                    'assets/sprites/redbird-midflap.jpg',
                    'assets/sprites/redbird-downflap.jpg',
                ),
            'blue':
                (
                    'assets/sprites/bluebird-upflap.jpg',
                    'assets/sprites/bluebird-midflap.jpg',
                    'assets/sprites/bluebird-downflap.jpg',
                ),
            'yellow':
                (
                    'assets/sprites/yellowbird-upflap.jpg',
                    'assets/sprites/yellowbird-midflap.jpg',
                    'assets/sprites/yellowbird-downflap.jpg',
                )
        }
        self.BACKGROUNDS_LIST = {
            'day': 'assets/sprites/background-day.jpg',
            'night': 'assets/sprites/background-night.jpg',
        }
        self.PIPE_LIST = {
            'green': 'assets/sprites/pipe-green.jpg',
            'red': 'assets/sprites/pipe-red.jpg',
        }
        from PIL import Image
        self.IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/1.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/2.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/3.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/4.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/5.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/6.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/7.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/8.jpg').convert_alpha(),
            pygame.image.load('assets/sprites/9.jpg').convert_alpha()
        )
        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.jpg').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load('assets/sprites/message.jpg').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load('assets/sprites/base.jpg').convert_alpha()

    def loadSound(self):
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
        if self.playery + playerh >= self.BASEY - 1 or self.playery < 0:
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

    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0  # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (self.SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()

def image_process():
    import os
    from PIL import Image
    path = 'assets/sprites/'
    pic_list = os.listdir(path)
    for pic in pic_list:
        if pic.endswith('png'):
            img = Image.open(os.path.join(path, pic)).convert('RGB')\
                .save(os.path.join(path, pic.replace('png', 'jpg')))
        print(pic)


if __name__ == '__main__':
    # image_process()
    action_space = [[1, 0], [0, 1], [1,0]]
    env = GameEnv()
    env.reset()
    while True:
        # next_state, reward, _ = env.step([1, 0])
        next_state, reward, _ = env.step(1)
        print(reward)
        env.render()
        if _:
            break
