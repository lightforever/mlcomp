import {Component, Inject, ViewChild} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {SpaceRun} from "../../models";
import {SpaceService} from "./space.service";
import {DagsComponent} from "../../dag/dags/dags.component";

@Component({
    selector: 'space-run-dialog',
    templateUrl: 'space-run-dialog.html',
})
export class SpaceRunDialogComponent {

    @ViewChild(DagsComponent) dags;
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
        this.service.run(this.data).subscribe(res=>{
            if(res.success){
                this.dialogRef.close();
            }
            else {
                this.error = res.error;
            }
        });
    }
}