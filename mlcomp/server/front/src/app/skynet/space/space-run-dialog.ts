import {Component, Inject, ViewChild} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {SpaceRun} from "../../models";
import {SpaceService} from "./space.service";
import {DagsComponent} from "../../dag/dags/dags.component";
import {Helpers} from "../../helpers";

@Component({
    selector: 'space-run-dialog',
    templateUrl: 'space-run-dialog.html',
})
export class SpaceRunDialogComponent {

    @ViewChild(DagsComponent) dags;
    @ViewChild('textarea') textarea;
    error: string;

    constructor(
        public dialogRef: MatDialogRef<SpaceRunDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: SpaceRun,
        protected service: SpaceService
    ) {

    }

    onNoClick(): void {
        this.dialogRef.close();
    }

    on_ok_click() {
        this.data.dag = this.dags.selected.id;
        this.service.run(this.data).subscribe(res => {
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