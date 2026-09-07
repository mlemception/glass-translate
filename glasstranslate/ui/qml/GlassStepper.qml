pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Templates as T

/*!
    GlassStepper - T.SpinBox (docs/GLASS_DESIGN.md §2.3) for real numbers: a
    "fill" text field plus two "fill" stepper buttons.  T.SpinBox itself is
    integer-only, so the public API is real-valued and scaled internally by
    10^decimals:

        number    current value (bind it to the bridge; never assigned here)
        minimum / maximum / step / decimals / suffix
        committed(number)  after Enter, focus loss or a step

    Keyboard Up/Down step, Enter commits, the field ring shows on activeFocus.
    `label` / `hint` come from the FormRow and feed Accessible.*.
*/
T.SpinBox {
    id: control

    property real number: 0
    property real minimum: 0
    property real maximum: 100
    property real step: 1
    property int decimals: 0
    property string suffix: ""
    property string label: ""
    property string hint: ""
    signal committed(real number)

    readonly property real factor: Math.pow(10, decimals)
    readonly property real current: value / factor
    readonly property real buttonSize: Math.max(26, fm.height + 4)
    readonly property color inkColor: (Theme.mode === "hc" && !enabled) ? palette.disabled.windowText : Theme.ink

    from: Math.round(minimum * factor)
    to: Math.round(maximum * factor)
    stepSize: Math.max(1, Math.round(step * factor))
    editable: true

    implicitWidth: Math.max(implicitBackgroundWidth + leftInset + rightInset,
                            implicitContentWidth + leftPadding + rightPadding, 180)
    implicitHeight: Math.max(Theme.controlH, fm.height + 2 * Theme.padY,
                             implicitContentHeight + topPadding + bottomPadding)
    leftPadding: 12
    rightPadding: 12 + 2 * (buttonSize + 4)
    topPadding: Theme.padY
    bottomPadding: Theme.padY
    font.family: Theme.family
    font.pointSize: Theme.fontBody
    focusPolicy: Qt.StrongFocus
    hoverEnabled: true
    opacity: enabled || Theme.mode === "hc" ? 1 : 0.45

    Accessible.role: Accessible.SpinBox
    Accessible.name: label
    Accessible.description: hint

    FontMetrics { id: fm; font: control.font }

    onNumberChanged: syncValue()
    onFactorChanged: syncValue()
    Component.onCompleted: syncValue()
    onValueModified: committed(current)

    function syncValue() {
        var v = Math.round(number * factor)
        if (v !== value)
            value = v
    }

    function commitText(text) {
        var v = valueFromText(text, locale)
        v = Math.max(from, Math.min(to, v))
        if (v !== value) {
            value = v
            committed(current)
        } else {
            input.text = textFromValue(value, locale)
        }
    }

    textFromValue: function (v, loc) {
        return (v / control.factor).toFixed(control.decimals) + control.suffix
    }
    valueFromText: function (text, loc) {
        var t = String(text)
        if (control.suffix.length > 0 && t.endsWith(control.suffix))
            t = t.slice(0, t.length - control.suffix.length)
        var n = parseFloat(t.replace(",", ".").trim())
        return isNaN(n) ? control.value : Math.round(n * control.factor)
    }
    validator: RegularExpressionValidator { regularExpression: /^-?\d*[.,]?\d*.*$/ }

    background: Item {
        implicitWidth: 180
        implicitHeight: Theme.controlH
        readonly property real fieldWidth: width - 2 * (control.buttonSize + 4)
        GlassShadow {
            width: parent.fieldWidth; height: parent.height
            radius: Theme.radiusControl
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.16)
        }
        GlassSurface {
            id: field
            width: parent.fieldWidth; height: parent.height
            variant: "fill"
            radius: Theme.radiusControl
            borderColor: control.enabled ? Theme.borderColor : control.palette.disabled.windowText
        }
        FocusRing { target: field; ringRadius: Theme.radiusControl; active: control.activeFocus }
    }

    contentItem: TextInput {
        id: input
        text: control.displayText
        font: control.font
        color: control.inkColor
        selectionColor: Qt.rgba(Theme.ink.r, Theme.ink.g, Theme.ink.b, 0.25)
        selectedTextColor: Theme.ink
        selectByMouse: true
        horizontalAlignment: Qt.AlignLeft
        verticalAlignment: Qt.AlignVCenter
        readOnly: !control.editable
        validator: control.validator
        inputMethodHints: Qt.ImhFormattedNumbersOnly
        clip: true
        onEditingFinished: control.commitText(text)
    }

    component StepButton: Item {
        id: stepButton
        property string glyph: "plus"
        property bool pressed: false
        property bool hovered: false
        width: control.buttonSize
        height: control.buttonSize
        y: (control.height - height) / 2
        onPressedChanged: {
            stepIn.stop()
            stepOut.stop()
            if (Theme.reduceMotion) {
                scale = 1
            } else if (pressed) {
                stepIn.start()
            } else {
                stepOut.start()
            }
        }
        NumberAnimation { id: stepIn; target: stepButton; property: "scale"; to: Theme.pressScale; duration: Theme.pressInMs; easing.type: Easing.OutCubic }
        SpringAnimation { id: stepOut; target: stepButton; property: "scale"; to: 1; spring: Theme.springPress.spring; damping: Theme.springPress.damping }
        GlassShadow {
            anchors.fill: parent
            radius: Theme.concentric(Theme.radiusControl, 2)
            sigma: 5
            offsetY: 2
            color: Qt.rgba(0, 0, 0, 0.16)
        }
        GlassSurface {
            anchors.fill: parent
            variant: "fill"
            radius: Theme.concentric(Theme.radiusControl, 2)
            fillColor: Theme.mode !== "hc" && stepButton.hovered && control.enabled
                       ? Qt.rgba(1, 1, 1, Math.min(1, Theme.fillControl.a + 0.06)) : Theme.fillControl
            borderColor: control.enabled ? Theme.borderColor : control.palette.disabled.windowText
        }
        Glyph {
            anchors.centerIn: parent
            name: stepButton.glyph
            color: control.inkColor
        }
    }

    up.indicator: StepButton {
        x: control.width - width
        glyph: "plus"
        pressed: control.up.pressed
        hovered: control.up.hovered
    }
    down.indicator: StepButton {
        x: control.width - 2 * width - 4
        glyph: "minus"
        pressed: control.down.pressed
        hovered: control.down.hovered
    }
}
