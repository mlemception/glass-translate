pragma ComponentBehavior: Bound
import QtQuick

/*!
    TitleBar - our own frameless title bar (docs/GLASS_DESIGN.md §1.1, §2.3):
    app name (title size, weight 600) with the 6 px accent running dot, and
    28 px "fill" minimise / close buttons.  A press on the bar emits
    `dragRequested()` synchronously so Main.qml can call bridge.startMove()
    inside the press (startSystemMove acts on the current button press).
    Double-click does nothing.
*/
Item {
    id: bar

    property string title: "GlassTranslate"
    property bool running: false
    signal dragRequested()
    signal minimizeRequested()
    signal closeRequested()

    implicitHeight: Math.max(Theme.titleH, titleText.implicitHeight + 8, 28 + 8)

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        onPressed: bar.dragRequested()
    }

    Row {
        id: heading
        anchors.left: parent.left
        anchors.leftMargin: 4
        anchors.verticalCenter: parent.verticalCenter
        spacing: 8
        Text {
            id: titleText
            text: bar.title
            font.family: Theme.family
            font.pointSize: Theme.fontTitle
            font.weight: 600
            color: Theme.ink
            anchors.verticalCenter: parent.verticalCenter
        }
        Rectangle {
            width: 6
            height: 6
            radius: 3
            color: Theme.accent
            visible: bar.running
            anchors.verticalCenter: parent.verticalCenter
            Accessible.role: Accessible.Indicator
            Accessible.name: "Running"
        }
    }

    Row {
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        spacing: 6
        GlassButton {
            objectName: "minimizeButton"
            width: 28
            height: 28
            radius: 8
            leftPadding: 0
            rightPadding: 0
            iconGlyph: "minimize"
            label: "Minimise"
            onClicked: bar.minimizeRequested()
        }
        GlassButton {
            objectName: "closeButton"
            width: 28
            height: 28
            radius: 8
            leftPadding: 0
            rightPadding: 0
            iconGlyph: "close"
            label: "Close"
            onClicked: bar.closeRequested()
        }
    }
}
