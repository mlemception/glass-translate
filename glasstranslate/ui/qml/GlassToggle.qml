pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Window
import QtQuick.Templates as T

/*!
    GlassToggle - T.Switch (docs/GLASS_DESIGN.md §2.3): a 44x26 capsule track
    (accent when on) and a 22 px "pill" knob that lenses a LOCAL
    ShaderEffectSource of the track beneath it.  The knob stretches 4 px while
    pressed and springs to the other side on release (Theme.springGrab; 120 ms
    OutCubic under reduceMotion); the whole indicator compresses to .97 while
    pressed.  `text` is the caption to the right; `label` / `hint` come from the
    FormRow and feed Accessible.*.
*/
T.Switch {
    id: control

    property string label: ""
    property string hint: ""

    readonly property real dpr: Screen.devicePixelRatio > 0 ? Screen.devicePixelRatio : 1
    readonly property color inkColor: (Theme.mode === "hc" && !enabled) ? palette.disabled.windowText : Theme.ink

    implicitWidth: Math.max(implicitBackgroundWidth + leftInset + rightInset,
                            implicitContentWidth + leftPadding + rightPadding)
    implicitHeight: Math.max(Theme.controlH,
                             implicitContentHeight + topPadding + bottomPadding,
                             implicitIndicatorHeight + topPadding + bottomPadding)
    padding: Theme.padY
    leftPadding: 2
    spacing: Theme.gap
    font.family: Theme.family
    font.pointSize: Theme.fontBody
    focusPolicy: Qt.StrongFocus
    hoverEnabled: true
    opacity: enabled || Theme.mode === "hc" ? 1 : 0.45

    Accessible.role: Accessible.CheckBox
    Accessible.name: label.length > 0 ? label : text
    Accessible.description: hint

    onDownChanged: {
        pressIn.stop()
        pressOut.stop()
        if (Theme.reduceMotion) {
            ind.scale = 1
        } else if (down) {
            pressIn.start()
        } else {
            pressOut.start()
        }
    }
    NumberAnimation { id: pressIn; target: ind; property: "scale"; to: 0.97; duration: Theme.pressInMs; easing.type: Easing.OutCubic }
    SpringAnimation { id: pressOut; target: ind; property: "scale"; to: 1; spring: Theme.springPress.spring; damping: Theme.springPress.damping }

    indicator: Item {
        id: ind
        implicitWidth: 44
        implicitHeight: 26
        x: control.leftPadding
        y: control.topPadding + (control.availableHeight - height) / 2

        // the layer the knob lenses: track only (the caption sits outside it)
        Item {
            id: trackLayer
            anchors.fill: parent
            GlassSurface {
                anchors.fill: parent
                variant: "card"
                radius: height / 2
                squircle: 2
                fillColor: control.checked ? Theme.accent : Theme.fillControl
                Behavior on fillColor { ColorAnimation { duration: Theme.fade } }
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
        GlassShadow {
            x: knob.x; y: knob.y; width: knob.width; height: knob.height
            radius: knob.radius
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.18)
        }
        GlassSurface {
            id: knob
            variant: "pill"
            height: 22
            width: control.down && control.enabled ? 26 : 22
            y: 2
            x: 2 + control.visualPosition * (ind.width - width - 4)
            radius: height / 2
            sourceItem: trackLayer
            sharpSource: trackSrc
            Behavior on width { NumberAnimation { duration: Theme.pressInMs; easing.type: Easing.OutCubic } }
            Behavior on x {
                enabled: !control.down
                animation: Theme.reduceMotion ? knobEase : knobSpring
            }
        }
        NumberAnimation { id: knobEase; duration: Theme.motionDuration; easing.type: Easing.OutCubic }
        SpringAnimation { id: knobSpring; spring: Theme.springGrab.spring; damping: Theme.springGrab.damping }
        FocusRing { ringRadius: ind.height / 2; active: control.visualFocus }
    }

    contentItem: Text {
        leftPadding: control.indicator ? control.indicator.width + control.spacing : 0
        text: control.text
        font: control.font
        color: control.inkColor
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
