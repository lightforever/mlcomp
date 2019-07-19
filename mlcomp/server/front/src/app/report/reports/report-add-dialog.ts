import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material";
import {ReportAddData} from "../../models";

@Component({
    selector: 'report-add-dialog',
    templateUrl: 'report-add-dialog.html',
})
export class ReportAddDialogComponent {

    constructor(
        public dialogRef: MatDialogRef<ReportAddDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data: ReportAddData) {

        if(data.projects.length>0){
            data.project = data.projects[0].id;
        }
        if(data.layouts.length>0){
            data.layout = data.layouts[0].id;
        }
    }

    onNoClick(): void {
        this.dialogRef.close();
    }

}