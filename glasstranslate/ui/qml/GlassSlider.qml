pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Window
import QtQuick.Templates as T

/*!
    GlassSlider - T.Slider (docs/GLASS_DESIGN.md §2.3): 4 px track with an accent
    fill and a 20 px "pill" knob lensing a LOCAL ShaderEffectSource of the
    track + fill layer beneath it.  On grab the knob swells to 1.28x with
    Theme.springGrab and settles back with Theme.springSwell (one visible
    bounce); the shadow grows with it.  `valueText` (e.g. "10 %") is drawn to
    the right.  `live` (inherited) makes `moved` fire on every step while
    dragging.
*/
T.Slider {
    id: control

    property string valueText: ""
    property string label: ""
    property string hint: ""

    readonly property real knobSize: 20
    readonly property real dpr: Screen.devicePixelRatio > 0 ? Screen.devicePixelRatio : 1
    readonly property color inkColor: (Theme.mode === "hc" && !enabled) ? palette.disabled.windowText : Theme.ink

    implicitWidth: 200
    implicitHeight: Math.max(Theme.controlH, knobSize * Theme.grabScale + 4, fm.height + 2 * Theme.padY)
    leftPadding: 0
    rightPadding: valueLabel.visible ? valueLabel.implicitWidth + Theme.gap : 0
    font.family: Theme.family
    font.pointSize: Theme.fontBody
    focusPolicy: Qt.StrongFocus
    hoverEnabled: true
    snapMode: T.Slider.SnapAlways
    opacity: enabled || Theme.mode === "hc" ? 1 : 0.45

    Accessible.role: Accessible.Slider
    Accessible.name: label
    Accessible.description: hint

    FontMetrics { id: fm; font: control.font }

    onPressedChanged: {
        grabAnim.stop()
        swellAnim.stop()
        if (Theme.reduceMotion) {
            knob.scale = 1
        } else if (pressed) {
            grabAnim.start()
        } else {
            swellAnim.start()
        }
    }
    SpringAnimation { id: grabAnim; target: knob; property: "scale"; to: Theme.grabScale; spring: Theme.springGrab.spring; damping: Theme.springGrab.damping }
    SpringAnimation { id: swellAnim; target: knob; property: "scale"; to: 1; spring: Theme.springSwell.spring; damping: Theme.springSwell.damping }

    background: Item {
        implicitWidth: 200
        implicitHeight: Theme.controlH

        // the layer the knob lenses: full travel width so the lens never samples past its edge
        Item {
            id: trackLayer
            x: control.leftPadding
            width: control.availableWidth
            height: parent.height
            Rectangle {
                id: groove
                x: control.knobSize / 2
                width: parent.width - control.knobSize
                height: 4
                radius: 2
                anchors.verticalCenter: parent.verticalCenter
                color: Theme.mode === "hc" ? Theme.sysPal.windowText : Theme.fillTrack
            }
            Rectangle {
                x: groove.x
                width: control.visualPosition * groove.width
                height: 4
                radius: 2
                anchors.verticalCenter: parent.verticalCenter
                color: Theme.accent
            }
        }
        ShaderEffectSource {
            id: trackSrc
            sourceItem: trackLayer
            hideSource: false
            live: true
            smooth: true
            textureSize: Qt.size(trackLayer.width * control.dpr, trackLayer.height * control.dpr)
        }
        Text {
            id: valueLabel
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            visible: control.valueText.length > 0
            text: control.valueText
            font: control.font
            color: control.inkColor
            horizontalAlignment: Text.AlignRight
        }
    }

    handle: Item {
        id: knob
        width: control.knobSize
        height: control.knobSize
        x: control.leftPadding + control.visualPosition * (control.availableWidth - width)
        y: control.topPadding + (control.availableHeight - height) / 2
        GlassShadow {
            anchors.fill: parent
            radius: width / 2
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.18)
        }
        GlassSurface {
            anchors.fill: parent
            variant: "pill"
            radius: width / 2
            sourceItem: trackLayer
            sharpSource: trackSrc
        }
        FocusRing { ringRadius: knob.width / 2; active: control.visualFocus }
    }
}
