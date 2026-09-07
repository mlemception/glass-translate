pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T

/*!
    GlassButton - T.Button with the "fill" material (docs/GLASS_DESIGN.md §2.3):
    soft shadow instead of a border, `primary` blends the accent at 22 % over the
    fill (Theme.fillPrimary), optional `iconGlyph` (Glyph name) before the text.
    Press: scale .965 in 90 ms OutCubic, released with Theme.springPress
    (nothing under reduceMotion).  Focus ring on keyboard focus only.
    `label` / `hint` come from the enclosing FormRow and feed Accessible.*.
*/
T.Button {
    id: control

    property bool primary: false
    property string iconGlyph: ""
    property string label: ""
    property string hint: ""
    property real radius: Theme.radiusControl

    implicitWidth: Math.max(implicitBackgroundWidth + leftInset + rightInset,
                            implicitContentWidth + leftPadding + rightPadding, 96)
    implicitHeight: Math.max(Theme.controlH, fm.height + 2 * Theme.padY,
                             implicitContentHeight + topPadding + bottomPadding)
    leftPadding: 14
    rightPadding: 14
    topPadding: Theme.padY
    bottomPadding: Theme.padY
    font.family: Theme.family
    font.pointSize: Theme.fontBody
    font.weight: primary ? 600 : 500
    focusPolicy: Qt.StrongFocus
    hoverEnabled: true
    opacity: enabled || Theme.mode === "hc" ? 1 : 0.45

    Accessible.role: Accessible.Button
    Accessible.name: text.length > 0 ? text : label
    Accessible.description: hint

    readonly property color inkColor: (Theme.mode === "hc" && !enabled) ? palette.disabled.windowText : Theme.ink
    property color fillTone: {
        var base = control.primary ? Theme.fillPrimary : Theme.fillControl
        if (Theme.mode === "hc")
            return base
        var lift = (control.hovered && control.enabled ? 0.06 : 0) + (control.checked && !control.primary ? 0.10 : 0)
        return Qt.rgba(base.r, base.g, base.b, Math.min(1, base.a + lift))
    }
    Behavior on fillTone { ColorAnimation { duration: Theme.motionDuration } }

    FontMetrics { id: fm; font: control.font }

    // press-compress: 90 ms OutCubic in, spring out (Theme.pressScale is 1 under reduceMotion)
    onDownChanged: {
        pressIn.stop()
        pressOut.stop()
        if (Theme.reduceMotion) {
            scale = 1
        } else if (down) {
            pressIn.start()
        } else {
            pressOut.start()
        }
    }
    NumberAnimation { id: pressIn; target: control; property: "scale"; to: Theme.pressScale; duration: Theme.pressInMs; easing.type: Easing.OutCubic }
    SpringAnimation { id: pressOut; target: control; property: "scale"; to: 1; spring: Theme.springPress.spring; damping: Theme.springPress.damping }

    background: Item {
        implicitWidth: 96
        implicitHeight: Theme.controlH
        GlassShadow {
            anchors.fill: parent
            radius: control.radius
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.16)
        }
        GlassSurface {
            anchors.fill: parent
            variant: "fill"
            radius: control.radius
            fillColor: control.fillTone
            borderColor: control.enabled ? Theme.borderColor : control.palette.disabled.windowText
        }
        FocusRing { ringRadius: control.radius; active: control.visualFocus }
    }

    contentItem: Item {
        implicitWidth: line.implicitWidth
        implicitHeight: line.implicitHeight
        Row {
            id: line
            anchors.centerIn: parent
            spacing: 6
            Glyph {
                visible: control.iconGlyph.length > 0
                name: control.iconGlyph
                color: control.inkColor
                anchors.verticalCenter: parent.verticalCenter
            }
            Text {
                visible: control.text.length > 0
                text: control.text
                font: control.font
                color: control.inkColor
                verticalAlignment: Text.AlignVCenter
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }
}
