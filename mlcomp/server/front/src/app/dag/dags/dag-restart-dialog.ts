import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {DagRestart} from "../../models";
import {DagService} from "../dag.service";

@Component({
    selector: 'dag-restart-dialog',
    templateUrl: 'dag-restart-dialog.html',
})
export class DagRestartDialogComponent {
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
        this.service.restart(this.data).subscribe(res=>{
            if(res.success){
                this.dialogRef.close();
            }
            else {
                this.error = res.error;
            }
        });
    }
}