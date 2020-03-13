import {Component, Inject, ViewChild} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {SpaceAdd} from "../../models";
import {SpaceService} from "./space.service";
import {Helpers} from "../../helpers";

@Component({
    selector: 'space-add-dialog',
    templateUrl: 'space-add-dialog.html',
})
export class SpaceAddDialogComponent {
    @ViewChild('textarea') textarea;
    error: string;

    constructor(
        public dialogRef: MatDialogRef<SpaceAddDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: SpaceAdd,
        protected service: SpaceService
    ) {

    }

    onNoClick(): void {
        this.dialogRef.close();
    }

    on_ok_click() {
        if (this.data.method == 'add') {
            this.service.add(this.data.space).subscribe(res => {
                this.error = res.error;
                if (res.success) {
                    this.dialogRef.close();
                }
            });
        } else if (this.data.method == 'edit') {
            this.service.edit(this.data.space).subscribe(res => {
                this.error = res.error;
                if (res.success) {
                    this.dialogRef.close();
                }
            });
        } else if (this.data.method == 'copy') {
            this.service.copy(this.data.space, this.data.old_space).subscribe(res => {
                this.error = res.error;
                if (res.success) {
                    this.dialogRef.close();
                }
            });
        }
    }

    key_down(event) {
        let content = Helpers.handle_textarea_down_key(event,
            this.textarea.nativeElement);
        if (content) {
            this.data.space.content = content;
        }

    }

    key_up(event) {
        this.data.space.content = event.target.value;
    }
}