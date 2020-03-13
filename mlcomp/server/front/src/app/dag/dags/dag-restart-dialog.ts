import {Component, Inject, ViewChild} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {DagRestart} from "../../models";
import {DagService} from "../dag.service";
import {Helpers} from "../../helpers";

@Component({
    selector: 'dag-restart-dialog',
    templateUrl: 'dag-restart-dialog.html',
})
export class DagRestartDialogComponent {
    @ViewChild('textarea') textarea;
    error: string;

    constructor(
        public dialogRef: MatDialogRef<DagRestartDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: DagRestart,
        protected service: DagService
    ) {

    }

    onNoClick(): void {
        this.dialogRef.close();
    }

    on_ok_click() {
        this.service.restart(this.data).subscribe(res => {
            if (res.success) {
                this.dialogRef.close();
            } else {
                this.error = res.error;
            }
        });
    }

    key_down(event) {
        let content = Helpers.handle_textarea_down_key(event,
            this.textarea.nativeElement);
        if (content) {
            this.data.file_changes = content;
        }

    }

    key_up(event) {
        this.data.file_changes = event.target.value;
    }
}