#!/usr/bin/env python

# Author: Justin Zortea
# Adapted from the Panda3D "looking-and-gripping" sample, by Shao Zhang and Phil Saltzman

from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import TextNode, NodePath, LightAttrib
from panda3d.core import LVector3
from direct.actor.Actor import Actor
from direct.task.Task import Task
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.DirectObject import DirectObject
import sys

import collections
import math

import cv2, time
import numpy as np

from displacement import displacement
from face import faces

# Macro-like function used to reduce the amount to code needed to create the
# on screen instructions
def genLabelText(text, i):
    return OnscreenText(text=text, parent=base.a2dTopLeft, scale=.06,
                        pos=(0.06, -.08 * i), fg=(1, 1, 1, 1),
                        shadow=(0, 0, 0, .5), align=TextNode.ALeft)

def addInstructions(pos, msg):
    return OnscreenText(text=msg, parent=base.a2dTopLeft,
                        style=1, fg=(1, 1, 1, 1), pos=(0.06, -pos - 0.03),
                        align=TextNode.ALeft, scale=.05)



# Downsample frames for faster processing, and normalize
def downsample(frame):
    FACTOR = 4
    return frame[::FACTOR, ::FACTOR].astype(np.float32)/255.


class WatchMe(ShowBase):

    def __init__(self):
        # Initialize the ShowBase class from which we inherit, which will
        # create a window and set up everything we need for rendering into it.
        ShowBase.__init__(self)

        # Put the standard title and instruction text on screen
        self.modeText = OnscreenText(text="Optical Flow",
                                  fg=(1, 1, 1, 1), parent=base.a2dBottomRight,
                                  align=TextNode.ARight, pos=(-0.1, 0.1),
                                  shadow=(0, 0, 0, .5), scale=.08)

        instr0 = addInstructions(0.06, "Press ESC to exit")
        instr1 = addInstructions(0.12, "Press 1: Facial Recognition")
        instr1 = addInstructions(0.18, "Press 2: Optical Flow")

        # Set up key input
        self.accept('escape', self.destroy)
        self.accept('1', self.chooseFacialRec)
        self.accept('2', self.chooseOpticalFlow)

        self.mode = 'opticalFlow'

        self.environ = loader.loadModel("models/world")
        self.environ.reparentTo(render)

        # Disable mouse-based camera-control
        base.disableMouse()  

        self.eve = Actor("models/eve")

        self.eve.setScale(.7)
        self.eve.setPos(5, 5, 3)

        # Put it in the scene
        self.eve.reparentTo(render)

        self.camDistance = 15
        self.camHeight = 5

        # Position the camera
        camera.setPos(self.eve.getPos() + (0, -1 * self.camDistance, self.camHeight))

        # Now we use controlJoint to get a NodePath that's in control of her neck
        # This must be done before any animations are played
        self.eveNeck = self.eve.controlJoint(None, 'modelRoot', 'Neck')

        self.camera.lookAt(self.eve)
        self.camBaseZ = self.camera.getZ()
        self.camBaseX = self.camera.getX()

        self.pitchOffset = self.camDistance
        self.headingOffset = self.camDistance

        # Now we add a task that will take care of turning the head
        taskMgr.add(self.turnHead, "turnHead")
       
        # Put in some default lighting
        self.setupLights()  

        self.video = cv2.VideoCapture(0)
        check, self.prev_frame = self.video.read()
        self.prev_frame = downsample(self.prev_frame)
        self.total_displacement = [0, 0]

        self.prev_face_x = None
        self.prev_face_y = None
        self.total_face_displacement = [0, 0]

        self.xQueueSize = 20
        self.xQueue = collections.deque([0, 0, 0, 0, 0], self.xQueueSize)
        self.xQueueSum = 0
        
        self.yQueueSize = 20
        self.yQueue = collections.deque([0, 0, 0, 0, 0], self.yQueueSize)
        self.yQueueSum = 0

    def updateModeText(self, text):
        self.modeText.setText(text)

    def chooseFacialRec(self):
        self.mode = 'face'
        self.updateModeText('Facial Recognition')

    def chooseOpticalFlow(self):
        self.mode = 'opticalFlow'
        self.updateModeText('Optical Flow')

    # Clean up
    def destroy(self):
        self.video.release()
        cv2.destroyAllWindows
        sys.exit

    # Move camera based on displacement
    # Turn eve's head to look at new camera position
    def turnHead(self, task):

        print(self.eve.getDistance(self.camera))
        
        # Move camera up and down relative to displacement 
        # between frames (scaled arbitrarily for good practical results)
        # Move eve's head to look at camera
        if self.mode == 'opticalFlow':
            self.getDisplacement()
            self.camera.setX(self.camBaseX + self.total_displacement[0] * -0.35)
            self.eveNeck.setP(self.getP())
            self.camera.setZ(self.camBaseZ + self.total_displacement[1] * -0.35)
            self.eveNeck.setH(self.getH())
        elif self.mode == 'face':
            self.getFaces()
            self.camera.setX(self.camBaseX + self.total_face_displacement[0] * -0.1)
            self.eveNeck.setP(self.getP())
            self.camera.setZ(self.camBaseZ + self.total_face_displacement[1] * -0.1)
            self.eveNeck.setH(self.getH())
        else:
            print('Unrecognized mode')

        # Rotate camera to center on eve
        self.camera.lookAt(self.eve)

        # Task continues infinitely
        return Task.cont  

    # Trig to get heading of eve's head, looking at camera
    def getH(self):
        A = self.pitchOffset
        O = self.camera.getZ() - self.camBaseZ + self.camHeight
        H = math.sqrt(A**2 + O**2)
        self.headingOffset = H
        return np.degrees(np.arctan(O / A))

    # Trig to get pitch of eve's head, looking at camera
    def getP(self):
        A = self.headingOffset
        O = self.camera.getX() - self.camBaseX
        H = math.sqrt(A**2 + O**2)
        self.headingOffset = H
        return np.degrees(np.arctan(O / A))

    def getFaces(self):
        check, frame = self.video.read()
        f = faces(frame)
        # Only use the first detected face
        if(len(f) > 0):
            x = f[0][0]
            y = f[0][1]
            w = f[0][2]
            h = f[0][3]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            if not self.prev_face_x is None:
                delta = self.prev_face_x - x
                minus = self.xQueue.popleft()
                self.xQueue.append(delta)
                self.xQueueSum -= minus
                self.xQueueSum += delta
                average = self.xQueueSum / self.xQueueSize
                self.total_face_displacement[0] -= average
            self.prev_face_x = x
            if not self.prev_face_y is None:
                delta = self.prev_face_y - y
                minus = self.yQueue.popleft()
                self.yQueue.append(delta)
                self.yQueueSum -= minus
                self.yQueueSum += delta
                average = self.yQueueSum / self.yQueueSize
                self.total_face_displacement[1] -= average
            self.prev_face_y = y

        cv2.imshow("Faces", frame)
        return f

    def getDisplacement(self):
        check, frame = self.video.read()
        frame = downsample(frame)
        # Compute displacement between current and previous frame
        d = displacement(np.array([self.prev_frame, frame]))
        self.total_displacement += d
        # print(self.total_displacement)
        self.prev_frame = frame
        cv2.imshow("Capturing", frame)
        return d

    # Sets up some default lighting
    def setupLights(self):
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.4, .4, .35, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(LVector3(0, 8, -2.5))
        directionalLight.setColor((0.9, 0.8, 0.9, 1))
        render.setLight(render.attachNewNode(directionalLight))
        render.setLight(render.attachNewNode(ambientLight))

demo = WatchMe()  # Create an instance of our class
demo.run()  # Run the simulation
